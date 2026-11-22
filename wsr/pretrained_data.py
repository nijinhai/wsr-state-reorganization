"""Document-preserving subsets of the original WikiText-103 and PG-19 splits."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request

import torch

ROOT=Path('artifacts/pretrained-v1/data')


def get_json(url):
    with urllib.request.urlopen(url,timeout=60) as response: return json.load(response)


def save_document(dataset,split,identity,text,tokenizer,source,cap=40000):
    encoded=tokenizer.encode(text,add_special_tokens=False).ids
    if len(encoded)<1025: return None
    path=ROOT/dataset/split/(str(identity)+'.pt')
    path.parent.mkdir(parents=True,exist_ok=True)
    metadata={'dataset':dataset,'split':split,'id':str(identity),'source':source,
              'sha256':hashlib.sha256(text.encode()).hexdigest(),
              'document_key':source.get('title',str(identity)).strip().casefold(),
              'original_tokens':len(encoded),'stored_tokens':min(len(encoded),cap),
              'truncated_prefix':len(encoded)>cap,'tokenizer':'EleutherAI/gpt-neox-20b'}
    if path.exists():
        old=torch.load(path,weights_only=True)
        if old['metadata']!=metadata: raise ValueError(f'Document changed: {path}')
    else:
        torch.save({'tokens':torch.tensor(encoded[:cap],dtype=torch.long),'metadata':metadata},path)
    return metadata


def wikitext(split,count,tokenizer,minimum_tokens=1025):
    documents=[]
    title=None
    lines=[]
    first_row=0
    for offset in range(0,5000,100):
        query=urllib.parse.urlencode({'dataset':'Salesforce/wikitext','config':'wikitext-103-raw-v1',
                                     'split':split,'offset':offset,'length':100})
        page=get_json('https://datasets-server.huggingface.co/rows?'+query)
        for row in page['rows']:
            text=row['row']['text']
            # Subsection markers contain additional '=' and do not start a new document.
            if re.fullmatch(r'\s*=\s+[^=]+?\s+=\s*',text):
                if title is not None:
                    joined=''.join(lines)
                    identity=str(first_row)+'-'+hashlib.sha256(title.encode()).hexdigest()[:12]
                    record=save_document('wikitext103',split,identity,joined,tokenizer,
                                         {'repository':'Salesforce/wikitext','config':'wikitext-103-raw-v1',
                                          'title':title,'start_row':first_row,'end_row_exclusive':row['row_idx']})
                    if record and record['stored_tokens']>=minimum_tokens: documents.append(record)
                    if len(documents)>=count: return documents
                title=text.strip();lines=[text];first_row=row['row_idx']
            elif title is not None:
                lines.append(text)
        print('WikiText',split,'rows',offset+100,'complete documents',len(documents),flush=True)
    raise RuntimeError('Not enough complete WikiText documents; do not fabricate document boundaries')


def pg19(split,count,tokenizer):
    # Use the official split membership list and download only selected individual books.
    url=f'https://huggingface.co/datasets/deepmind/pg19/resolve/main/data/{split}_files.txt'
    with urllib.request.urlopen(url,timeout=60) as response: names=sorted(response.read().decode().splitlines())
    documents=[]
    for name in names[:50]:
        url='https://storage.googleapis.com/deepmind-gutenberg/'+name
        with urllib.request.urlopen(url,timeout=60) as response: text=response.read().decode('utf-8')
        if len(tokenizer.encode(text,add_special_tokens=False).ids)<35000: continue
        identity=Path(name).stem
        record=save_document('pg19',split,identity,text,tokenizer,
                             {'url':url,'official_split_list_sha256':hashlib.sha256('\n'.join(names).encode()).hexdigest()})
        documents.append(record)
        print('PG19',split,identity,record['stored_tokens'],flush=True)
        if len(documents)>=count: return documents
    raise RuntimeError('Insufficient long books in selected official split prefix')


def documents(dataset,split):
    result=[]
    for path in sorted((ROOT/dataset/split).glob('*.pt')):
        record=torch.load(path,weights_only=True)
        if record['metadata']['split']!=split: raise ValueError('Split mismatch')
        result.append(record)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--splits',nargs='+',choices=['train','validation','test'],default=['train','validation'])
    p.add_argument('--final-test',action='store_true')
    args=p.parse_args()
    if 'test' in args.splits and not args.final_test:
        p.error('Test preparation requires explicit --final-test; default work uses train/validation only')
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file('artifacts/pretrained-v1/model/tokenizer.json')
    ROOT.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for split in args.splits:
        manifest+=wikitext(split,12 if split=='train' else 2,tokenizer,minimum_tokens=1025 if split=='train' else 4096)
        manifest+=pg19(split,4 if split=='train' else 2,tokenizer)
    # Protect both document identity and exact duplicates across the original splits.
    existing=[]
    for path in ROOT.glob('*/*/*.pt'):
        existing.append(torch.load(path,weights_only=True)['metadata'])
    for i,a in enumerate(existing):
        for b in existing[i+1:]:
            if a['split']!=b['split'] and (a['sha256']==b['sha256'] or (a['dataset']==b['dataset'] and a['document_key']==b['document_key'])):
                raise ValueError('Cross-split document leakage')
    (ROOT/'manifest.json').write_text(json.dumps(existing,indent=2))
    print('Prepared',len(existing),'split-preserving documents; test untouched unless explicitly requested.',flush=True)

if __name__=='__main__': main()
