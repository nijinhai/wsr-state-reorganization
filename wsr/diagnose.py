"""Base-only controlled diagnostics. Does not import or construct Dream."""
import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F

from .model import Backbone, State
from .tasks import QUERY, SEP, READ, world

class GRUBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(40,32)
        self.gru = nn.GRU(32,32,batch_first=True)
        self.output = nn.Linear(32,40)

    def run(self,tokens,state=None):
        output,state=self.gru(self.embedding(tokens),state)
        return self.output(output),state

@dataclass(frozen=True)
class Profile:
    family: str
    length: int = 12
    episodes: int = 1
    pairs: int = 8
    distractors: bool = False


def samples(seed,profile):
    if profile.family in ('memory','redundancy','rules'):
        return world(seed,profile.family,episodes=profile.episodes,length=profile.length)
    g=torch.Generator().manual_seed(seed)
    answer=int(torch.randint(8,(),generator=g))
    if profile.family=='copy':
        filler=(torch.randint(8,24,(profile.length,),generator=g).tolist()
                if profile.distractors else [36]*profile.length)
        tokens=[answer]+filler+[QUERY]
    elif profile.family=='lookup':
        aliases=torch.randperm(16,generator=g)[:profile.pairs]+8
        labels=torch.randperm(8,generator=g)[:profile.pairs]
        query=int(torch.randint(profile.pairs,(),generator=g))
        tokens=[]
        for alias,label in zip(aliases.tolist(),labels.tolist()):
            tokens.extend([alias,label,SEP])
        tokens.extend([int(aliases[query]),QUERY])
        answer=int(labels[query])
    else:
        raise ValueError(profile)
    return [(torch.tensor([tokens]),torch.tensor([answer]))]


def batch(seeds,profile):
    worlds=[samples(seed,profile) for seed in seeds]
    return [(torch.cat([w[i][0] for w in worlds]),torch.cat([w[i][1] for w in worlds]))
            for i in range(profile.episodes)]


def read(model,state,target):
    output,_=model.run(torch.full((target.shape[0],1),READ,dtype=torch.long,device=target.device),state)
    return F.cross_entropy(output[:,0],target),output[:,0]


def objective(model,data):
    state=None
    losses=[]
    logits=[]
    for tokens,target in data:
        _,state=model.run(tokens,state)
        loss,prediction=read(model,state,target)
        losses.append(loss)
        logits.append(prediction)
    return torch.stack(losses).mean(),torch.cat(logits),torch.cat([y for _,y in data])


@torch.no_grad()
def evaluate(model,data):
    loss,logits,target=objective(model,data)
    counts=Counter(target.tolist())
    return {'loss':float(loss),'accuracy':float((logits.argmax(-1)==target).float().mean()),
            'n':len(target),'label_counts':dict(sorted(counts.items())),
            'uniform_8':0.125,'majority':max(counts.values())/len(target)}


def make_model(kind,seed=0):
    torch.manual_seed(seed)
    model=GRUBackbone() if kind=='gru' else Backbone()
    if kind=='ssm_slow':
        # Controlled timescale initialization ablation, all equations unchanged.
        with torch.no_grad():
            model.log_a.fill_(-3.0)
    return model


def fit(kind,profile,steps,fixed=False,seed=0,checkpoint=None):
    model=make_model(kind,seed)
    optimizer=torch.optim.Adam(model.parameters(),lr=0.002)
    fixed_data=batch(range(20000,20032),profile)
    test_data=batch(range(1000000,1000128),profile)
    result={'torch_version':str(torch.__version__),'kind':kind,'seed':seed,'profile':asdict(profile),'steps':steps,'fixed32':fixed,
            'parameters':sum(p.numel() for p in model.parameters()),'initial':evaluate(model,fixed_data),'trace':[]}
    start=time.monotonic()
    for step in range(steps):
        data=fixed_data if fixed else batch(range(200000+step*16,200000+(step+1)*16),profile)
        optimizer.zero_grad(set_to_none=True)
        loss,_,_=objective(model,data)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss: {kind} {profile} step={step}')
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
        optimizer.step()
        if step==0 or (step+1)%50==0 or step==steps-1:
            result['trace'].append({'step':step+1,'loss':float(loss.detach()),'gradient_norm':float(norm)})
        if fixed and (step+1)%50==0 and evaluate(model,fixed_data)['accuracy']>=0.999:
            result['actual_steps']=step+1
            break
    result['actual_steps']=result.get('actual_steps',steps)
    result['train' if fixed else 'reference32']=evaluate(model,fixed_data)
    result['test']=evaluate(model,test_data)
    result['seconds']=time.monotonic()-start
    if checkpoint:
        torch.save({'model':model.state_dict(),'diagnosis':result,'optimizer':optimizer.state_dict()},checkpoint)
    return result


def oracle(episodes,family):
    answers=[]
    total=0
    for tokens,_ in episodes:
        values=tokens[0].tolist()
        assert values[-1]==QUERY
        if family=='memory':
            total=(total+sum(values[:-1]))%8
            answers.append(total)
        elif family=='redundancy':
            mapping={}
            for i in range(0,len(values)-2,3):
                alias,label,separator=values[i:i+3]
                assert separator==SEP
                assert alias not in mapping
                mapping[alias]=label
            assert len(mapping)==16
            answers.append(mapping[values[-2]])
        elif family=='rules':
            mapping={values[i]:values[i+1] for i in range(0,24,3)}
            answers.append(mapping[mapping[values[-3]]])
    return answers


def chain_check(model,profile):
    model.zero_grad(set_to_none=True)
    data=batch(range(100,104),profile)
    before={name:p.detach().clone() for name,p in model.named_parameters()}
    loss,logits,_=objective(model,data)
    loss.backward()
    grads={name:float(p.grad.norm()) if p.grad is not None else None for name,p in model.named_parameters()}
    optimizer=torch.optim.Adam(model.parameters(),lr=0.002)
    optimizer.step()
    updates={name:float((p.detach()-before[name]).norm()) for name,p in model.named_parameters()}
    return {'loss':float(loss.detach()),'logits_finite':bool(torch.isfinite(logits).all()),
            'gradient_norms':grads,'update_norms':updates,
            'parameters_finite':all(bool(torch.isfinite(p).all()) for p in model.parameters()),
            'dtype':str(next(model.parameters()).dtype)}


def retention(model,tokens):
    state=model.initial(tokens.shape[0],tokens.device)
    decays=[]
    states=[]
    for token in tokens.unbind(1):
        e=model.embedding(token)
        window=torch.cat([state.conv,e[:,None]],1)
        x=F.silu((window*model.conv_weight).sum(1))
        parameters=model.select(x).reshape(-1,model.heads,2*model.memory+1)
        dt=F.softplus(parameters[...,-1])
        decays.append(torch.exp(-dt*model.log_a.exp()))
        output,state=model.run(token[:,None],state)
        state.ssm.retain_grad()
        states.append(state.ssm)
    output[:,-1].square().mean().backward()
    decay=torch.stack(decays,1).detach()
    return {'decay_mean_per_head':decay.mean((0,1)).tolist(),
            'half_life_tokens':(math.log(.5)/decay.log().mean((0,1))).tolist(),
            'first_state_survival_to_read':decay[:,1:].double().prod(1).flatten().tolist(),
            'state_gradient_norm_per_token':[float(s.grad.double().norm()) for s in states],
            'max_abs_state':max(float(s.detach().abs().max()) for s in states)}


def audit(directory):
    result={'task_examples':{},'oracle_checks':{},'chain':{},'retention':{}}
    for family in ['memory','redundancy','rules']:
        episodes=world(42,family,episodes=3)
        result['task_examples'][family]=[{'episode':i,'tokens':x[0].tolist(),'label':int(y),
            'oracle':answer} for i,((x,y),answer) in enumerate(zip(episodes,oracle(episodes,family)))]
        count=Counter()
        checks=0
        for seed in range(500):
            data=world(seed,family,episodes=3)
            answers=oracle(data,family)
            assert answers==[int(y) for _,y in data]
            count.update(answers)
            checks+=len(answers)
        result['oracle_checks'][family]={'checked':checks,'mismatches':0,'label_counts':dict(sorted(count.items())),
            'uniform_answer_random':0.125,'uniform_vocab_random':0.025,'empirical_majority':max(count.values())/checks}
    for kind in ['ssm','gru']:
        result['chain'][kind]=chain_check(make_model(kind),Profile('redundancy',episodes=3))
    for label in ['fresh','original_checkpoint']:
        model=make_model('ssm')
        if label=='original_checkpoint':
            model.load_state_dict(torch.load('artifacts/base.pt',weights_only=True)['model'])
        for family in ['memory','redundancy']:
            tokens=torch.cat([x for x,_ in world(42,family,episodes=3)],1)
            tokens=torch.cat([tokens,torch.tensor([[READ]])],1)
            result['retention'][f'{label}_{family}']=retention(model,tokens)
    (directory/'audit.json').write_text(json.dumps(result,indent=2))
    print('audit: all 4500 oracle answers match; chain checks saved',flush=True)



def protocol(directory,steps=2000):
    """Reproduce the original one-world/step, mixed-family, three-episode protocol."""
    model=make_model('gru')
    optimizer=torch.optim.Adam(model.parameters(),lr=0.002)
    trace=[]
    start=time.monotonic()
    for step in range(steps):
        data=world(step,['memory','redundancy'][step%2],episodes=3)
        optimizer.zero_grad(set_to_none=True)
        loss,_,_=objective(model,data)
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
        optimizer.step()
        if (step+1)%200==0:
            trace.append({'step':step+1,'loss':float(loss.detach()),'gradient_norm':float(norm)})
            print('matched GRU protocol',step+1,flush=True)
    result={'training_steps':steps,'seed':0,'trace':trace,'seconds':time.monotonic()-start,'evaluations':{}}
    original=make_model('ssm')
    original.load_state_dict(torch.load('artifacts/base.pt',weights_only=True)['model'])
    for kind,network in [('ssm_original_2000',original),('gru',model)]:
        result['evaluations'][kind]={}
        for family in ['memory','redundancy','rules']:
            result['evaluations'][kind][family]=evaluate(network,batch(range(1000000,1000030),Profile(family,episodes=3)))
    torch.save({'model':model.state_dict(),'diagnosis':result},directory/'gru-original-protocol.pt')
    (directory/'protocol.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result['evaluations']),flush=True)


def course(directory,kind,family,steps=900):
    # Same optimizer budget and world seeds; compare a graded schedule to direct challenge training.
    stages=([Profile('memory',length=n) for n in [2,4,6,12]] if family=='memory' else
            [Profile('lookup',pairs=n) for n in [1,2,4]]+[Profile('redundancy')])
    results=[]
    for schedule in ['direct','curriculum']:
        model=make_model(kind)
        optimizer=torch.optim.Adam(model.parameters(),lr=0.002)
        trace=[]
        start=time.monotonic()
        for step in range(steps):
            profile=stages[min(3,step*4//steps)] if schedule=='curriculum' else stages[-1]
            optimizer.zero_grad(set_to_none=True)
            loss,_,_=objective(model,batch(range(500000+step*16,500000+(step+1)*16),profile))
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
            optimizer.step()
            if (step+1)%(steps//4)==0:
                trace.append({'step':step+1,'profile':asdict(profile),'loss':float(loss.detach()),
                              'gradient_norm':float(norm)})
        result={'kind':kind,'family':family,'schedule':schedule,'steps':steps,'trace':trace,
                'test':[{**asdict(p),**evaluate(model,batch(range(1000000,1000128),p))} for p in stages],
                'seconds':time.monotonic()-start}
        results.append(result)
        name=f'{kind}-{family}-{schedule}'
        torch.save({'model':model.state_dict(),'diagnosis':result},directory/(name+'.pt'))
        (directory/f'course-{kind}-{family}.json').write_text(json.dumps(results,indent=2))
        print(name,[(r['family'],r['length'],r['pairs'],round(r['accuracy'],3)) for r in result['test']],flush=True)



@torch.no_grad()
def state_ablation(model,data):
    rows={}
    tokens,target=data[0]
    _,state=model.run(tokens)
    for label,ssm,conv in [('intact',state.ssm,state.conv),
                           ('zero_ssm',torch.zeros_like(state.ssm),state.conv),
                           ('zero_conv',state.ssm,torch.zeros_like(state.conv))]:
        loss,logits=read(model,State(ssm,conv),target)
        rows[label]={'accuracy':float((logits.argmax(-1)==target).float().mean()),'loss':float(loss)}
    return rows


def controls(directory,steps=300):
    results=[]
    for seed in [0,1,2]:
        for kind in ['ssm','ssm_slow']:
            profile=Profile('copy',length=24)
            path=directory/f'readwrite-{kind}-seed-{seed}.pt'
            result=fit(kind,profile,steps,seed=seed,checkpoint=path)
            model=make_model(kind,seed)
            model.load_state_dict(torch.load(path,weights_only=True)['model'])
            result['state_ablation']=state_ablation(model,batch(range(1000000,1000128),profile))
            result['transfer']=[{**asdict(p),**evaluate(model,batch(range(1000000,1000128),p))}
                                for p in [Profile('copy',length=n) for n in [0,2,8,24,48]]]
            results.append(result)
            (directory/'controls.json').write_text(json.dumps(results,indent=2))
            print('timescale control',seed,kind,result['test']['accuracy'],result['state_ablation'],flush=True)
    checkpoint=torch.load('artifacts/base.pt',weights_only=True)
    fp32=make_model('ssm')
    fp32.load_state_dict(checkpoint['model'])
    fp64=make_model('ssm').double()
    fp64.load_state_dict(checkpoint['model'])
    precision={}
    with torch.no_grad():
        for family in ['memory','redundancy']:
            data=batch(range(1000000,1000030),Profile(family,episodes=3))
            _,a,_=objective(fp32,data)
            _,b,_=objective(fp64,data)
            precision[family]={'max_logits_difference':float((a.double()-b).abs().max()),
                               'prediction_agreement':float((a.argmax(-1)==b.argmax(-1)).float().mean())}
    (directory/'precision.json').write_text(json.dumps(precision,indent=2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['audit','overfit','ladder','protocol','curriculum','controls'])
    p.add_argument('--directory',type=Path,default=Path('artifacts/diagnosis-v1'))
    p.add_argument('--steps',type=int,default=300)
    p.add_argument('--family',choices=['memory','redundancy'],default='memory')
    p.add_argument('--kinds',nargs='+',choices=['ssm','ssm_slow','gru'],default=['ssm','gru'])
    args=p.parse_args()
    if args.steps<1 or (args.phase=='curriculum' and args.steps%4):
        p.error('--steps must be positive; curriculum requires a multiple of four')
    torch.set_num_threads(1)
    args.directory.mkdir(parents=True,exist_ok=True)
    reserved={'audit':['audit.json'],'protocol':['protocol.json'],'controls':['controls.json'],
              'curriculum':[f'course-{k}-{args.family}.json' for k in args.kinds],
              'overfit':['overfit.json'],'ladder':['ladder.json']}[args.phase]
    for name in reserved:
        if (args.directory/name).exists():
            raise FileExistsError(f'Preserve previous diagnostics: {args.directory/name}; choose a new --directory')
    if args.phase=='controls':
        controls(args.directory,args.steps)
        return
    if args.phase=='protocol':
        protocol(args.directory,args.steps)
        return
    if args.phase=='curriculum':
        for kind in args.kinds:
            course(args.directory,kind,args.family,args.steps)
        return
    if args.phase=='audit':
        audit(args.directory)
        return
    profiles=([Profile('memory'),Profile('redundancy')] if args.phase=='overfit' else
              [Profile('copy',length=n) for n in [0,2,8,24]]+
              [Profile('copy',length=8,distractors=True)]+
              [Profile('memory',length=n) for n in [2,4,12]]+
              [Profile('lookup',pairs=n) for n in [1,2,4]]+
              [Profile('redundancy'),Profile('memory',episodes=3),Profile('redundancy',episodes=3)])
    output=args.directory/(args.phase+'.json')
    if output.exists():
        raise FileExistsError(f'Preserve previous results: {output}')
    results=[]
    for profile in profiles:
        for kind in args.kinds:
            label=f'{kind}-{profile.family}-{profile.length}-{profile.episodes}-{profile.pairs}-{profile.distractors}'
            result=fit(kind,profile,args.steps,fixed=args.phase=='overfit',checkpoint=args.directory/(label+'-'+args.phase+'.pt'))
            results.append(result)
            output.write_text(json.dumps(results,indent=2))
            print(label,'fit32=' if args.phase=='overfit' else 'reference32=',round(result['train' if args.phase=='overfit' else 'reference32']['accuracy'],3),'test=',round(result['test']['accuracy'],3),
                  'steps=',result['actual_steps'],'sec=',round(result['seconds'],1),flush=True)

if __name__=='__main__':
    main()
