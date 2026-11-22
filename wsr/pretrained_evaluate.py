"""Paired real-text evaluation, with post-edit logits and uninterrupted book state."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .pretrained_model import Cache,LayerCache
from .pretrained_data import documents
from .pretrained_experiment import ROOT,model_load,load_dream
from .state_dream import edit_cache,trace_metrics


def seed_for(record,event):
    return int(hashlib.sha256(record['metadata']['sha256'].encode()).hexdigest()[:7],16)+event


def method_load(name,seed,device):
    if name in ['baseline','corrupt']:
        return None,{'layer':12,'history_window':128,'corruption':name=='corrupt','rounds':8}
    suffix='-state-only' if name.endswith('-state-only') else ''
    case=name.split('-')[0]
    dream,config=load_dream(ROOT/'formal'/f'{case}-seed-{seed}{suffix}',device)
    return dream,config


def serialize_cache(cache):
    return {'processed':cache.processed,'layers':[{'conv':s.conv.cpu(),'ssm':s.ssm.cpu()} for s in cache.layers]}


def deserialize_cache(record,device):
    return Cache(tuple(LayerCache(s['conv'].to(device),s['ssm'].to(device)) for s in record['layers']),record['processed'])


@torch.no_grad()
def prefix_caches(model,record,boundaries):
    identity=record['metadata']['dataset']+'-'+record['metadata']['split']+'-'+record['metadata']['id']
    directory=ROOT/'prefix-cache'/identity
    directory.mkdir(parents=True,exist_ok=True)
    state=None
    tokens=record['tokens'][None].to(next(model.parameters()).device)
    result={}
    for boundary in sorted(boundaries):
        path=directory/f'{boundary}.pt'
        if path.exists():
            saved=torch.load(path,weights_only=True)
            if saved['text_sha256']!=record['metadata']['sha256'] or saved['model_revision']!=model.asset_manifest['revision']:raise ValueError('Stale prefix cache')
            state=deserialize_cache(saved,next(model.parameters()).device)
        else:
            start=0 if state is None else state.processed
            state=model.wake(tokens[:,start:boundary-1],state)
            saved=serialize_cache(state);saved['text_sha256']=record['metadata']['sha256']
            saved['model_revision']=model.asset_manifest['revision']
            torch.save(saved,path)
        result[boundary]=state
    return result


@torch.no_grad()
def score(model,tokens,cache,boundary,length,chunk=256):
    if cache.processed!=boundary-1:raise ValueError('Score would reuse stale logits')
    values=[]
    for start in range(0,length,chunk):
        end=min(length,start+chunk)
        logits,cache=model.run(tokens[:,boundary-1+start:boundary-1+end],cache)
        targets=tokens[:,boundary+start:boundary+end]
        loss=F.cross_entropy(logits.flatten(0,1),targets.flatten(),reduction='none')
        values.extend(loss.double().cpu().tolist())
    return values,cache


def loss_summary(values):
    bins={}
    previous=0
    for end in [32,128,512,2048]:
        stop=min(len(values),end)
        if stop>previous:
            bins[f'{previous+1}-{stop}']=sum(values[previous:stop])/(stop-previous)
        previous=end
    cumulative={str(n):sum(values[:n])/n for n in [32,128,512,2048] if len(values)>=n}
    return {'tokens':len(values),'nll':sum(values)/len(values),'distance_bins':bins,'cumulative_horizons':cumulative}


@torch.no_grad()
def single(model,name,seed,kind,output):
    dream,config=method_load(name,seed,next(model.parameters()).device)
    dataset='wikitext103' if kind=='short' else 'pg19'
    pool=documents(dataset,'validation')
    records=([r for r in pool if len(r['tokens'])>=4096] if kind=='short' else pool)[:2]
    if len(records)<2: raise ValueError('Prepare two complete eligible validation documents; never join documents to reach a length')
    boundaries=[512,2048] if kind=='short' else [8192,32768]
    rows=[]
    for record in records:
        available=[b for b in boundaries if b+512<=len(record['tokens'])]
        states=prefix_caches(model,record,available)
        tokens=record['tokens'][None].to(next(model.parameters()).device)
        for boundary in available:
            cache=states[boundary]
            changed,trace,initial=edit_cache(model,dream,cache,tokens,boundary,config['history_window'],config['layer'],
                                              corruption=config['corruption'],seed=seed_for(record,boundary))
            length=min(2048,len(record['tokens'])-boundary)
            values,_=score(model,tokens,changed,boundary,length)
            row={'document':record['metadata'],'history_visible':boundary,'state_history':boundary-1,
                 'history_window':config['history_window'],'losses':values,**loss_summary(values),
                 'edit_trace':trace_metrics(initial,trace),
                 'corruption_relative_norm':float((initial-cache.layers[config['layer']].ssm).norm()/cache.layers[config['layer']].ssm.norm()),
                 'total_relative_norm':float((changed.layers[config['layer']].ssm-cache.layers[config['layer']].ssm).norm()/cache.layers[config['layer']].ssm.norm())}
            rows.append(row)
            if trace:
                torch.save({'original':cache.layers[config['layer']].ssm.cpu(),'input':initial.cpu(),
                            'states':[x.cpu() for x in trace]},output/f'trajectory-{record["metadata"]["id"]}-{boundary}.pt')
            print(name,seed,kind,record['metadata']['id'],'history',boundary,'nll',round(row['nll'],7),flush=True)
    return rows


@torch.no_grad()
def periodic(model,name,seed,output,total=32768,interval=2048):
    dream,config=method_load(name,seed,next(model.parameters()).device)
    record=documents('pg19','validation')[0]
    tokens=record['tokens'][None].to(next(model.parameters()).device)
    state=prefix_caches(model,record,[512])[512]
    boundary=512;event=0;rows=[];all_losses=[]
    while boundary<min(total,len(record['tokens'])):
        changed,trace,initial=edit_cache(model,dream,state,tokens,boundary,config['history_window'],config['layer'],
                                          corruption=config['corruption'],seed=seed_for(record,boundary))
        length=min(interval,total-boundary,len(record['tokens'])-boundary)
        old_ssm=state.layers[config['layer']].ssm
        corruption_relative=float((initial-old_ssm).norm()/old_ssm.norm())
        total_relative=float((changed.layers[config['layer']].ssm-old_ssm).norm()/old_ssm.norm())
        values,state=score(model,tokens,changed,boundary,length)
        row={'event':event,'history_visible':boundary,'state_history':boundary-1,
             'history_window':config['history_window'],'losses':values,**loss_summary(values),
             'edit_trace':trace_metrics(initial,trace),'ending_state_history':state.processed,
             'corruption_relative_norm':corruption_relative,'total_relative_norm':total_relative}
        rows.append(row);all_losses+=values
        if event==0 and trace:
            torch.save({'input':initial.cpu(),'states':[x.cpu() for x in trace]},output/'periodic-first-trajectory.pt')
        boundary+=length;event+=1
        print(name,seed,'periodic','event',event,'through',boundary,'nll',round(row['nll'],7),flush=True)
    # Interventions never reset the cache, even after their effect becomes harmful.
    return {'document':record['metadata'],'interval':interval,'total_visible_tokens':boundary,
            'cache_resets_within_book':0,'events':rows,'mean_nll':sum(all_losses)/len(all_losses),
            'stream_prefix_nll':{str(n):sum(all_losses[:n-512])/(n-512) for n in [8192,32768] if len(all_losses)>=n-512}}


@torch.no_grad()
def rounds(model,name,seed,output):
    dream,config=method_load(name,seed,next(model.parameters()).device)
    if dream is None:raise ValueError('Round sweep needs a trained Dream')
    record=documents('wikitext103','validation')[0]
    tokens=record['tokens'][None].to(next(model.parameters()).device)
    boundary=512;cache=prefix_caches(model,record,[boundary])[boundary]
    changed,trace,initial=edit_cache(model,dream,cache,tokens,boundary,config['history_window'],config['layer'],
                                     corruption=config['corruption'],seed=seed_for(record,boundary))
    rows=[]
    for r,h in enumerate([initial]+trace):
        losses,_=score(model,tokens,cache.edit(config['layer'],h),boundary,512)
        rows.append({'round':r,**loss_summary(losses)})
    torch.save({'initial':initial.cpu(),'states':[h.cpu() for h in trace]},output/'round-states.pt')
    return {'document':record['metadata'],'history_visible':boundary,'rows':rows,
            'trace':trace_metrics(initial,trace),'note':'Same model trained for eight rounds; early-round readouts are iteration ablations, not separately trained one-shot controls.'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('kind',choices=['short','long-single','periodic','rounds'])
    p.add_argument('--name',default='baseline')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    output=ROOT/'evaluation'/f'{args.name}-seed-{args.seed}'
    output.mkdir(parents=True,exist_ok=True)
    path=output/(args.kind+'.json')
    if path.exists():print('Preserved completed evaluation',path);return
    model=model_load(args.device)
    start=time.monotonic()
    if args.kind in ['short','long-single']:result=single(model,args.name,args.seed,args.kind,output)
    elif args.kind=='periodic':result=periodic(model,args.name,args.seed,output)
    else:result=rounds(model,args.name,args.seed,output)
    path.write_text(json.dumps({'name':args.name,'seed':args.seed,'kind':args.kind,'split':'validation',
                                'result':result,'seconds':time.monotonic()-start},indent=2))

if __name__=='__main__':main()
