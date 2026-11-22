"""Paired B0/S/P training with explicit budgets, frozen-prefix reuse, and resumable checkpoints."""
import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from .pretrained_experiment import model_load,continuation
from .pretrained_evaluate import deserialize_cache,score
from .state_dream import StateDream,edit_cache,history_input
from .v02_aux import QueryDecoder,FeaturePredictor,TargetBank,TRAIN_QUERIES,query_tensors
from .v02_data import ROOT,LAYER


def write_json(path,data):
    temp=Path(str(path)+'.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(path)


def save_checkpoint(path,data):
    temp=Path(str(path)+'.tmp');torch.save(data,temp);temp.replace(path)


def schedule(rows,seed,steps=2048):
    pools={name:[r for r in rows if r['document']['dataset']==name] for name in ['wikitext103','pg19']}
    for name,values in pools.items():random.Random(seed*10000+len(values)).shuffle(values)
    return [pools[['wikitext103','pg19'][i%2]][(i//2)%len(pools[['wikitext103','pg19'][i%2]])] for i in range(steps)]


class Inputs:
    def __init__(self,model):
        self.model=model;self.device=next(model.parameters()).device
        self.index=json.loads((ROOT/'bank/index.json').read_text());self.tokens={}
        for row in self.index['train']+self.index['validation']:
            if row['token_path'] not in self.tokens:
                self.tokens[row['token_path']]=torch.load(row['token_path'],weights_only=True)['tokens'][None].to(self.device)
        docs=sorted(set(r['document']['id'] for r in self.index['validation'] if r['document']['dataset']=='wikitext103'))[:4]
        self.curve=[r for r in self.index['validation'] if (r['document']['dataset']=='wikitext103' and r['document']['id'] in docs and r['boundary']==512)
                    or (r['document']['dataset']=='pg19' and r['boundary']==8192)]
        self.curve_states={r['id']:self.load(r) for r in self.curve}

    def load(self,row):
        data=torch.load(row['path'],weights_only=True,mmap=True)
        if data['document']['sha256']!=row['document']['sha256']:raise ValueError('Cache document mismatch')
        if data['model_revision']!=self.model.asset_manifest['revision']:raise ValueError('Cache backbone mismatch')
        return deserialize_cache(data,self.device)

    def original_second_state(self,row):
        data=torch.load(row['path'],weights_only=True,mmap=True)
        return data['cycle1_original_ssm'].to(self.device)


@torch.no_grad()
def validate(model,dream,inputs):
    rows=[]
    for row in inputs.curve:
        tokens=inputs.tokens[row['token_path']];state=inputs.curve_states[row['id']]
        edited,trace,h=edit_cache(model,dream,state,tokens,row['boundary'],128,LAYER)
        values,_=score(model,tokens,edited,row['boundary'],512)
        rows.append({'id':row['id'],'dataset':row['document']['dataset'],'nll':sum(values)/len(values),
                     'edit_relative':0. if not trace else float((trace[-1]-h).norm()/h.norm()),'tokens':len(values)})
    return {'mean_nll':sum(r['nll'] for r in rows)/len(rows),'rows':rows}


def load_run(path,device):
    saved=torch.load(path,map_location=device,weights_only=True)
    dream=StateDream(mode='D').to(device);dream.load_state_dict(saved['dream'])
    head=QueryDecoder().to(device) if saved['group']=='S' else FeaturePredictor().to(device) if saved['group']=='P' else None
    if head is not None:head.load_state_dict(saved['head'])
    return dream,head,saved


def gradient_stats(loss_a,loss_b,parameters):
    a=torch.autograd.grad(loss_a,parameters,retain_graph=True,allow_unused=True)
    b=torch.autograd.grad(loss_b,parameters,retain_graph=True,allow_unused=True)
    dot=sum((x*y).sum() for x,y in zip(a,b) if x is not None and y is not None)
    an=sum(x.square().sum() for x in a if x is not None).sqrt()
    bn=sum(x.square().sum() for x in b if x is not None).sqrt()
    return {'LM':float(an),'auxiliary_unweighted':float(bn),'cosine':float(dot/(an*bn).clamp_min(1e-20))}


def _train(group,seed,model,inputs,steps=2048):
    directory=ROOT/'runs'/f'{group}-seed-{seed}';directory.mkdir(parents=True,exist_ok=True)
    last=directory/'last.pt';device=inputs.device
    torch.manual_seed(seed)
    dream=StateDream(mode='D').to(device)
    # Heads are created after Dream, preserving exact paired Dream initialization.
    head=QueryDecoder().to(device) if group=='S' else FeaturePredictor().to(device) if group=='P' else None
    parameters=list(dream.parameters())+([] if head is None else list(head.parameters()))
    optimizer=torch.optim.Adam(parameters,lr=3e-4)
    control_head=None if head is None else copy.deepcopy(head)
    control_optimizer=None if head is None else torch.optim.Adam(control_head.parameters(),lr=3e-4)
    weight=0. if group=='B0' else json.loads((ROOT/'calibration.json').read_text())['weights'][group]
    rows=[];curve=[];done=0
    if last.exists():
        checkpoint=torch.load(last,map_location=device,weights_only=True)
        dream.load_state_dict(checkpoint['dream']);optimizer.load_state_dict(checkpoint['optimizer'])
        if head is not None:head.load_state_dict(checkpoint['head'])
        if control_head is not None:
            control_head.load_state_dict(checkpoint['control_head']);control_optimizer.load_state_dict(checkpoint['control_optimizer'])
        rows=checkpoint['training'];curve=checkpoint['validation'];done=checkpoint['step']
    if done>=steps and (directory/'summary.json').exists():print('Preserved complete run',group,seed,done,flush=True);return
    order=schedule(inputs.index['train'],seed)
    schedule_path=directory/'schedule.json'
    if not schedule_path.exists():write_json(schedule_path,[r['id'] for r in order])
    s_data=torch.load(ROOT/'aux-s/train.pt',weights_only=True,mmap=True) if group=='S' else None
    p_data=TargetBank('train',device) if group=='P' else None
    versions=[p._version for p in model.parameters()]
    original={k:v.detach().cpu().clone() for k,v in dream.state_dict().items()} if done==0 else None
    if not curve:
        baseline_path=ROOT/'no-dream-curve.json'
        baseline=json.loads(baseline_path.read_text()) if baseline_path.exists() else validate(model,None,inputs)
        write_json(baseline_path,baseline)
        curve.append({'step':0,**validate(model,dream,inputs)})
        save_checkpoint(directory/'checkpoint-0000.pt',{'group':group,'seed':seed,'step':0,
            'dream':dream.state_dict(),'head':None if head is None else head.state_dict(),
            'control_head':None if control_head is None else control_head.state_dict(),'auxiliary_weight':weight})
    torch.cuda.reset_peak_memory_stats(device) if device.type=='cuda' else None
    peak=0 if not last.exists() else checkpoint.get('peak_memory_bytes',0)
    start=time.monotonic()
    for step in range(done,steps):
        tick=time.monotonic();row=order[step];tokens=inputs.tokens[row['token_path']];state=inputs.load(row)
        boundary=row['boundary'];optimizer.zero_grad(set_to_none=True)
        losses=[];aux_losses=[];control_losses=[];edits=[]
        if control_optimizer is not None:control_optimizer.zero_grad(set_to_none=True)
        for cycle in range(2):
            edited,trace,initial=edit_cache(model,dream,state,tokens,boundary,128,LAYER)
            loss,advanced=continuation(model,tokens,edited,boundary,64);losses.append(loss)
            if group=='P':
                features,labels=p_data.targets(row,cycle,random.Random(seed*1000000+step*10+cycle))
                prediction=head(edited.layers[LAYER].ssm,torch.tensor([[0,1,2]],device=device))
                logits=torch.einsum('bdh,bdkh->bdk',prediction,features)/.1
                aux_losses.append(F.cross_entropy(logits.flatten(0,1),labels.flatten()))
                raw=initial if cycle==0 else inputs.original_second_state(row)
                control_prediction=control_head(raw.detach(),torch.tensor([[0,1,2]],device=device))
                control_logits=torch.einsum('bdh,bdkh->bdk',control_prediction,features)/.1
                control_losses.append(F.cross_entropy(control_logits.flatten(0,1),labels.flatten()))
            if step%64==0:edits.append(float((trace[-1]-initial).detach().norm()/initial.norm()))
            state=advanced.detach()
            if cycle==0:
                boundary+=256;state=model.wake(tokens[:,state.processed:boundary-1],state)
        if group=='S':
            example=s_data[random.Random(seed*100000+step).randrange(len(s_data))]
            raw=example['state'].to(device);visible=example['tokens'].to(device)
            history=history_input(model,visible,visible.shape[1],128)
            shared,_=dream(raw,history)
            queries,labels=query_tensors(example['rule'],TRAIN_QUERIES,device)
            logits=head(shared,queries)
            aux_losses.append(F.cross_entropy(logits.flatten(0,1),labels.flatten()))
            control_logits=control_head(raw.detach(),queries)
            control_losses.append(F.cross_entropy(control_logits.flatten(0,1),labels.flatten()))
        lm=torch.stack(losses).mean();aux=torch.stack(aux_losses).mean() if aux_losses else lm.new_zeros(())
        objective=lm+weight*aux
        if not torch.isfinite(objective):raise RuntimeError('Nonfinite loss')
        components=gradient_stats(lm,aux,list(dream.parameters())) if group!='B0' and step%256==0 else None
        objective.backward()
        dream_gradient=torch.nn.utils.clip_grad_norm_(dream.parameters(),1.,error_if_nonfinite=True)
        if not dream_gradient>0:raise RuntimeError('Dream received no gradient')
        if head is not None:torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        control_loss=torch.stack(control_losses).mean() if control_losses else lm.new_zeros(())
        if control_optimizer is not None:
            (weight*control_loss).backward();torch.nn.utils.clip_grad_norm_(control_head.parameters(),1.,error_if_nonfinite=True)
            control_optimizer.step()
        assert all(p.grad is None and not p.requires_grad for p in model.parameters())
        assert [p._version for p in model.parameters()]==versions
        metrics={'step':step+1,'id':row['id'],'document':row['document']['id'],'dataset':row['document']['dataset'],
                 'boundary':row['boundary'],'lm_nll':float(lm.detach()),'auxiliary':float(aux.detach()),
                 'dream_gradient_norm':float(dream_gradient),'seconds':time.monotonic()-tick,
                 'control_head_loss':float(control_loss.detach()),'auxiliary_weight':weight,
                 'actual_LM_forward_tokens':320,'LM_target_tokens':128,'effective_LM_batch_documents':1,
                 'cycles':2,'S_queries':len(TRAIN_QUERIES) if group=='S' else 0,
                 'P_queries':6 if group=='P' else 0,'gradient_components':components,'edit_relative':edits}
        rows.append(metrics)
        if step+1 in [32,128,256,512,768,1024,1536,2048]:
            curve.append({'step':step+1,**validate(model,dream,inputs)})
            write_json(directory/'validation.json',curve)
        if (step+1)%64==0 or step+1==steps:
            peak=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0
            checkpoint={'group':group,'seed':seed,'step':step+1,'dream':dream.state_dict(),
                        'head':None if head is None else head.state_dict(),'optimizer':optimizer.state_dict(),
                        'control_head':None if control_head is None else control_head.state_dict(),
                        'control_optimizer':None if control_optimizer is None else control_optimizer.state_dict(),
                        'auxiliary_weight':weight,
                        'training':rows,'validation':curve,'protocol':json.loads((ROOT/'protocol.json').read_text()),
                        'peak_memory_bytes':peak}
            save_checkpoint(last,checkpoint)
            if step+1 in [512,1024,2048]:save_checkpoint(directory/f'checkpoint-{step+1:04d}.pt',checkpoint)
            write_json(directory/'training.json',rows)
            print(group,seed,step+1,'LM',round(metrics['lm_nll'],5),'aux',round(metrics['auxiliary'],5),
                  'recent_sec/update',round(sum(r['seconds'] for r in rows[-64:])/min(64,len(rows)),3),flush=True)
    initial_weights=torch.load(directory/'checkpoint-0000.pt',map_location='cpu',weights_only=True)['dream']
    update=sum((dream.state_dict()[k].detach().cpu()-v).square().sum() for k,v in initial_weights.items()).sqrt().item()
    unique_docs=set((r['dataset'],r['document']) for r in rows)
    summary={'group':group,'seed':seed,'updates':len(rows),'LM_supervised_tokens':128*len(rows),
             'actual_LM_forward_tokens':320*len(rows),'unique_train_documents':len(unique_docs),
             'unique_prefixes':len(set(r['id'] for r in rows)),'effective_batch_documents':1,'cycles_per_update':2,
             'LM_target_tokens_per_update':128,'gradient_horizon':64,'dtype':'FP32','base_frozen':True,
             'nonzero_finite_Dream_gradients':all(r['dream_gradient_norm']>0 for r in rows),
             'dream_parameter_update_norm_this_invocation':update,
             'training_seconds_sum':sum(r['seconds'] for r in rows),'wall_seconds_this_invocation':time.monotonic()-start,
             'LM_target_tokens_per_second':128*len(rows)/sum(r['seconds'] for r in rows),
             'peak_allocated_bytes':peak,'auxiliary_head_parameters':0 if head is None else sum(p.numel() for p in head.parameters()),
             'final_validation':curve[-1],'complete':True}
    write_json(directory/'summary.json',summary)


def train(group,seed,model,inputs,steps=2048):
    scope=ROOT/'scope-update.json'
    if group=='S' and scope.exists():
        print('S training paused by scope update; existing results retained',seed,flush=True);return
    directory=ROOT/'runs'/f'{group}-seed-{seed}';directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _train(group,seed,model,inputs,steps)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--groups',nargs='+',choices=['B0','S','P'],default=['B0'])
    p.add_argument('--seeds',nargs='+',type=int,default=[0]);p.add_argument('--steps',type=int,default=2048)
    p.add_argument('--device',default='cuda');a=p.parse_args()
    model=model_load(a.device);inputs=Inputs(model)
    for seed in a.seeds:
        for group in a.groups:train(group,seed,model,inputs,a.steps)
