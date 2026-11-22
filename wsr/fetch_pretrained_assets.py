"""Download pinned public model assets without altering legacy experiments."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request

MODEL='state-spaces/mamba2-130m'
REVISION='3a5aea0c25d0fb43cc360e2c2aac82c26e3eed49'
TOKENIZER_REVISION='c292233c833e336628618a88a648727eb3dff0a7'


def fetch(url,path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): return path
    part=path.with_suffix(path.suffix+'.part')
    for attempt in range(8):
        offset=part.stat().st_size if part.exists() else 0
        request=urllib.request.Request(url,headers={'Range':f'bytes={offset}-'} if offset else {})
        try:
            with urllib.request.urlopen(request,timeout=60) as response:
                resumed=response.status==206 and offset>0
                mode='ab' if resumed else 'wb'
                count=offset if resumed else 0
                with part.open(mode) as output:
                    while True:
                        data=response.read(4*1024*1024)
                        if not data: break
                        output.write(data)
                        count+=len(data)
                        if count//(32*1024*1024)!=(count-len(data))//(32*1024*1024):
                            print(path.name,count,flush=True)
            part.rename(path)
            return path
        except Exception as e:
            print(path.name,'retry',attempt+1,type(e).__name__,flush=True)
            if attempt==7: raise
            time.sleep(2)


def main():
    root=Path('artifacts/pretrained-v1/model')
    api=json.load(urllib.request.urlopen(f'https://huggingface.co/api/models/{MODEL}/tree/{REVISION}',timeout=30))
    expected=next(x for x in api if x['path']=='pytorch_model.bin')
    metadata={'model':MODEL,'revision':REVISION,'expected_file':expected,'files':{}}
    for name in ['config.json','pytorch_model.bin']:
        path=fetch(f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}?download=true',root/name)
        digest=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
        if name=='pytorch_model.bin':
            assert path.stat().st_size==expected['size']
            assert digest==expected['lfs']['oid']
        metadata['files'][name]={'sha256':digest,'bytes':path.stat().st_size}
        (root/'manifest.json').write_text(json.dumps(metadata,indent=2))
    rev=TOKENIZER_REVISION
    path=fetch(f'https://huggingface.co/EleutherAI/gpt-neox-20b/resolve/{rev}/tokenizer.json',root/'tokenizer.json')
    metadata['tokenizer']={'model':'EleutherAI/gpt-neox-20b','revision':rev,'sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest()}
    (root/'manifest.json').write_text(json.dumps(metadata,indent=2))
    source_files={'mamba2.py':'mamba_ssm/modules/mamba2.py','ssd_minimal.py':'mamba_ssm/modules/ssd_minimal.py',
                  'layernorm_gated.py':'mamba_ssm/ops/triton/layernorm_gated.py','block.py':'mamba_ssm/modules/block.py',
                  'mixer_seq_simple.py':'mamba_ssm/models/mixer_seq_simple.py','LICENSE':'LICENSE'}
    vendor=root.parent/'vendor'
    sources={}
    for name,source in source_files.items():
        path=fetch('https://raw.githubusercontent.com/state-spaces/mamba/v2.2.5/'+source,vendor/name)
        sources[name]={'tag':'v2.2.5','path':source,'sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest()}
    (vendor/'manifest.json').write_text(json.dumps(sources,indent=2))
    print('Assets verified',flush=True)

if __name__=='__main__': main()
