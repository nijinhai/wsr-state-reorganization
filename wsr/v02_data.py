"""WSR-v0.2a document expansion and immutable, FP32 frozen-prefix banks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import urllib.parse

import torch

from . import pretrained_data as old
from .pretrained_experiment import model_load
from .pretrained_evaluate import serialize_cache

ROOT=Path('artifacts/wsr-v0.2a')
LAYER=12
DISTANCES=(128,512,2048)


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


def initialize():
    ROOT.mkdir(parents=True,exist_ok=True)
    path=ROOT/'v01-preservation.json'
    if not path.exists():
        files={str(p):digest(p) for p in Path('artifacts/pretrained-v1').rglob('*') if p.is_file()}
        path.write_text(json.dumps(files,indent=2))
        for p in Path('wsr').glob('*.py'):
            target=ROOT/'source-before'/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    protocol={'base':'official frozen state-spaces/mamba2-130m, pinned v0.1 revision',
              'architecture':'D Direct Dream, original StateDream; no workspace or memory slots',
              'layer':12,'rounds':8,'history_window':128,'corruption':False,'training':'T1',
              'prediction_horizon':64,'cycles':2,'interval':256,'batch_documents':1,
              'effective_LM_targets_per_update':128,'precision':'FP32','lr':0.0003,
              'milestones':[512,1024,2048],'seeds':[0,1,2],
              'auxiliary_weights':'B0=0; S/P separately calibrated from initial Dream-gradient norms and train-only smoke trials; see calibration.json',
              'P_distances':list(DISTANCES),'P_segment_tokens':64,'P_candidates':8,'P_temperature':0.1,
              'train_documents_requested':{'wikitext103':128,'pg19':16},
              'validation_wiki_requested':16,'test_split':'sealed',
              'seed_expansion_gate':'finite nonzero gradients, actual Dream updates, finite held-out NLL; not positive-NLL filtering',
              'equal_budget':'same optimizer steps, LM document/boundary schedule, target tokens, and seeds; auxiliary compute is additional and measured',
              'initialization':'fresh matched Dream initialization, not resumed from the v0.1 small-pool checkpoint',
              'BPTT':'64-token future in each cycle; detach between cycles, preserve state values'}
    dest=ROOT/'protocol.json'
    if dest.exists() and json.loads(dest.read_text())!=protocol:raise ValueError('Protocol changed')
    dest.write_text(json.dumps(protocol,indent=2))


def expanded_wiki(split,count,tokenizer):
    """Read original parquet rows in order; preserve complete article boundaries."""
    import urllib.request
    import pyarrow.parquet as pq
    directory=ROOT/'raw';directory.mkdir(parents=True,exist_ok=True)
    filename=f'{split}-00000-of-0000{2 if split=="train" else 1}.parquet'
    parquet=directory/filename
    metadata=old.get_json('https://huggingface.co/api/datasets/Salesforce/wikitext')
    revision=metadata['sha']
    if not parquet.exists():
        url=f'https://huggingface.co/datasets/Salesforce/wikitext/resolve/{revision}/wikitext-103-raw-v1/{filename}'
        with urllib.request.urlopen(url,timeout=120) as response,open(str(parquet)+'.part','wb') as target:shutil.copyfileobj(response,target)
        Path(str(parquet)+'.part').rename(parquet)
        (directory/(filename+'.json')).write_text(json.dumps({'url':url,'revision':revision,'sha256':digest(parquet)},indent=2))
    found=[];title=None;lines=[];first=0
    offset=0
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=1000,columns=['text']):
        for text in batch.column(0).to_pylist():
                row={'row_idx':offset};offset+=1
                if re.fullmatch(r'\s*=\s+[^=]+?\s+=\s*',text):
                    if title is not None:
                        identity=str(first)+'-'+hashlib.sha256(title.encode()).hexdigest()[:12]
                        r=old.save_document('wikitext103',split,identity,''.join(lines),tokenizer,
                                            {'repository':'Salesforce/wikitext','config':'wikitext-103-raw-v1',
                                             'title':title,'start_row':first,'end_row_exclusive':row['row_idx']})
                        if r and r['stored_tokens']>=3000:found.append(r)
                        if len(found)>=count:return found
                    title=text.strip();lines=[text];first=row['row_idx']
                elif title is not None:lines.append(text)
        print('expanded wiki',split,'rows',offset,'eligible',len(found),flush=True)
    raise RuntimeError('Not enough complete original documents')


def prepare_documents():
    from tokenizers import Tokenizer
    initialize()
    marker=ROOT/'data/manifest.json'
    if marker.exists():print('Preserved expanded corpus');return
    old.ROOT=ROOT/'data'
    tokenizer=Tokenizer.from_file('artifacts/pretrained-v1/model/tokenizer.json')
    selected=expanded_wiki('train',128,tokenizer)+old.pg19('train',16,tokenizer)
    selected+=expanded_wiki('validation',16,tokenizer)
    # Retain precisely the original two held-out PG books for long-range comparisons.
    for p in Path('artifacts/pretrained-v1/data/pg19/validation').glob('*.pt'):
        target=ROOT/'data/pg19/validation'/p.name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,target);selected.append(torch.load(p,weights_only=True)['metadata'])
    records=[]
    for p in (ROOT/'data').glob('*/*/*.pt'):
        r=torch.load(p,weights_only=True)['metadata'];records.append(r)
    for a in records:
        for b in records:
            if a['split']!=b['split'] and (a['sha256']==b['sha256'] or (a['dataset']==b['dataset'] and a['document_key']==b['document_key'])):
                raise ValueError('Cross-split leakage')
    marker.write_text(json.dumps({'selected':selected,'all_documents':records,'test_used':False},indent=2))
    print('selected documents',len(selected),'stored',len(records),flush=True)


def records(split):
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    return [(ROOT/'data'/r['dataset']/split/(r['id']+'.pt'),r) for r in manifest['selected'] if r['split']==split]


@torch.no_grad()
def prepare_bank(device='cuda'):
    model=model_load(device)
    index=[];val=[]
    for split in ['train','validation']:
        for n,(path,meta) in enumerate(records(split)):
            record=torch.load(path,weights_only=True);tokens=record['tokens'][None].to(device)
            rng=random.Random(int(meta['sha256'][:12],16))
            if split=='train':
                high=min(2048,tokens.shape[1]-2369)
                if high<512:continue
                boundaries=sorted(set([512,high]+[rng.randint(512,high) for _ in range(4)]))
            elif meta['dataset']=='wikitext103':
                boundaries=[b for b in [512,2048] if b+512<=tokens.shape[1]]
            else:boundaries=[512,8192,32768]
            state=None
            for boundary in boundaries:
                identity=f"{meta['dataset']}-{meta['id']}-{boundary}"
                dest=ROOT/'bank'/split/(identity+'.pt');dest.parent.mkdir(parents=True,exist_ok=True)
                if not dest.exists():
                    state=model.wake(tokens[:,0 if state is None else state.processed:boundary-1],state)
                    data=serialize_cache(state)
                    # Independent no-Dream feature-head control at the second visible boundary.
                    if split=='train':
                        control=model.wake(tokens[:,boundary-1:boundary+255],state)
                        data['cycle1_original_ssm']=control.layers[LAYER].ssm.cpu()
                    data.update({'document':meta,'boundary':boundary,'token_path':str(path),
                                 'model_revision':model.asset_manifest['revision']})
                    torch.save(data,dest)
                else:state=None # next missing state can always be reconstructed from the natural prefix
                row={'id':identity,'path':str(dest),'token_path':str(path),'boundary':boundary,'document':meta}
                (index if split=='train' else val).append(row)
            print('prefix bank',split,n+1,meta['id'],len(boundaries),flush=True)
    (ROOT/'bank/index.json').write_text(json.dumps({'train':index,'validation':val,'precision':'FP32'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['documents','bank']);p.add_argument('--device',default='cuda')
    a=p.parse_args()
    prepare_documents() if a.phase=='documents' else prepare_bank(a.device)
