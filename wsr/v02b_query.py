"""Final-answer-only query experiments. Original Mamba/Dream code is read-only."""
import argparse
import itertools
import json
from pathlib import Path
import random
import shutil
import time

import torch
from torch import nn
from torch.nn import functional as F

from .v02_aux import ALL_QUERIES, TRAIN_QUERIES, HELDOUT_QUERIES, generate_rules, answer
from .v02_data import digest
from .v02_train import write_json, save_checkpoint

ROOT=Path('artifacts/wsr-v0.2b')
OLD=Path('artifacts/wsr-v0.2a-diagnostics')
MILESTONES=[0,32,128,256,512,1024,2048,4096,8192]


def rule_key(rule):return tuple(rule[0]),tuple(rule[1])


def balanced_rules():
    original=generate_rules();used={rule_key(r) for values in original.values() for r in values}
    result=[]
    for mapping in itertools.permutations(range(4)):
        node=0;visited=[]
        for _ in range(4):visited.append(node);node=mapping[node]
        if len(set(visited))!=4 or node!=0:continue
        for colors in itertools.permutations(range(4)):
            rule=(mapping,colors)
            if rule_key(rule) not in used:result.append(rule)
    return result


def initialize():
    ROOT.mkdir(parents=True,exist_ok=True)
    before=ROOT/'preservation-before.json'
    if not before.exists():
        files=list(Path('wsr').glob('*.py'))+list(Path('tests').glob('*.py'))
        files=[p for p in files if 'v02b' not in p.name]
        files+=list(Path('artifacts/wsr-v0.2a/runs').glob('*/*.pt'))
        for root in [OLD,Path('artifacts/wsr-v0.2a')]:
            files+=list(root.glob('*.json'))+list(root.glob('*.md'))
        files+=list(OLD.glob('*head*.pt'))
        write_json(before,{str(p):digest(p) for p in files})
    protocol={'date':'2026-11-17','objective':'diagnose final-answer-only query composition; no language-model or Dream training',
              'original_train_rules':1024,'original_validation_rules':256,
              'train_queries':TRAIN_QUERIES,'heldout_queries':HELDOUT_QUERIES,
              'balanced_diagnostic_rules':len(balanced_rules()),
              'balanced_definition':'four-cycle mapping, four distinct colors, disjoint from original train and validation; never used for optimization',
              'batch_rules':16,'queries_per_rule':8,'main_updates':8192,
              'main_lr':.001,'optimizer':'Adam','gradient_clip':1.,'milestones':MILESTONES,
              'stage1':'original padded128 MLP64: overfit32, full8192; lr controls .0003/.003; capacity compact MLP128',
              'frozen':['Mamba','Direct Dream','original StateReader'],
              'future_scope':'minimal shared recurrent query, oracle learning gate before frozen-reader conditions',
              'no_W1_W2_or_S_plus_P':True,'test_split_used':False}
    path=ROOT/'protocol.json';normalized=json.loads(json.dumps(protocol))
    if path.exists():assert json.loads(path.read_text())==normalized
    else:write_json(path,protocol)


def oracle_features(facts,padded=True):
    values=F.one_hot(facts.long(),4).float().flatten(1)
    return F.pad(values,(0,96)) if padded else values


def labels_for(facts):
    return torch.tensor([[answer((f[:4],f[4:]),q) for q in ALL_QUERIES] for f in facts.tolist()])


def load_data():
    rules=generate_rules();rules['balanced']=balanced_rules();out={}
    cache=torch.load(OLD/'S-features.pt',weights_only=True,mmap=True)
    for split,values in rules.items():
        facts=torch.tensor([list(r[0])+list(r[1]) for r in values])
        block={'facts':facts,'labels':labels_for(facts),'oracle':oracle_features(facts),
               'rules':[rule_key(r) for r in values]}
        if split in cache:
            assert torch.equal(cache[split]['facts'],facts)
            block.update({k:cache[split]['features'][k] for k in ['raw_reader','edited_reader']})
        out[split]=block
    if (ROOT/'balanced-readers.pt').exists():
        extra=torch.load(ROOT/'balanced-readers.pt',weights_only=True)
        assert torch.equal(extra['facts'],out['balanced']['facts'])
        out['balanced'].update({k:extra[k] for k in ['raw_reader','edited_reader']})
    return out


class QueryMLP(nn.Module):
    def __init__(self,compact=False,hidden=64,depth=1):
        super().__init__();self.compact=compact
        self.entity=nn.Embedding(4,8);self.hops=nn.Embedding(3,8)
        layers=[nn.Linear((32 if compact else 128)+16,hidden),nn.SiLU()]
        for _ in range(depth-1):layers.extend([nn.Linear(hidden,hidden),nn.SiLU()])
        layers.append(nn.Linear(hidden,4));self.output=nn.Sequential(*layers)

    def forward(self,source,queries,**kwargs):
        if self.compact:source=source[:,:32]
        z=source[:,None].expand(-1,queries.shape[1],-1)
        query=torch.cat([self.entity(queries//3),self.hops(queries%3)],-1)
        return self.output(torch.cat([z,query],-1)),None


class RecurrentQuery(nn.Module):
    """Learned fact reading and GRU update; query supplies the execution depth.

    Entity-indexed input slots and a shared start/key embedding are structural
    priors. No link-following matrix multiplication or intermediate labels.
    """
    def __init__(self,source='oracle',hidden=16,loop_mode='query'):
        super().__init__();self.source=source;self.hidden=hidden;self.loop_mode=loop_mode
        self.node=nn.Embedding(4,hidden);self.hops=nn.Embedding(3,hidden)
        nn.init.normal_(self.hops.weight,std=.02)
        self.fact_value=nn.Sequential(nn.LayerNorm(8),nn.Linear(8,hidden),nn.SiLU())
        self.update=nn.GRUCell(hidden,hidden)
        self.output=nn.Sequential(nn.Linear(hidden*2,32),nn.SiLU(),nn.Linear(32,4))
        # Created last so core initialization is paired across source conditions.
        self.adapter=nn.Linear(128,32) if source!='oracle' else None

    def forward(self,source,queries,forced_rounds=None,intervention=None,replacement=None):
        flat=source[:,:32] if self.adapter is None else self.adapter(source)
        slots=torch.cat([flat[:,:16].reshape(-1,4,4),flat[:,16:].reshape(-1,4,4)],-1)
        values=self.fact_value(slots)
        w=self.node(queries//3)+self.hops(queries%3)
        initial=w;states=[];attention=[]
        depth=queries%3+1
        if self.loop_mode!='query':depth=torch.full_like(depth,int(self.loop_mode))
        if forced_rounds is not None:depth=torch.full_like(depth,forced_rounds)
        def read(work):
            weights=(torch.einsum('bqd,nd->bqn',work,self.node.weight)/self.hidden**.5).softmax(-1)
            return torch.einsum('bqn,bnd->bqd',weights,values),weights
        for step in range(3):
            if step==1 and intervention is not None:
                if intervention=='zero':changed=torch.zeros_like(w)
                elif intervention=='swap_rules':changed=w.roll(1,0)
                elif intervention=='reverse_first_update':changed=2*initial-w
                elif intervention=='replace':
                    assert replacement is not None and replacement.shape==w.shape
                    changed=replacement
                else:raise ValueError(intervention)
                w=torch.where((depth>1)[...,None],changed,w)
            evidence,weights=read(w)
            proposed=self.update(evidence.flatten(0,1),w.flatten(0,1)).reshape_as(w)
            w=torch.where((step<depth)[...,None],proposed,w)
            states.append(w);attention.append(weights)
        final_evidence,final_attention=read(w)
        logits=self.output(torch.cat([w,final_evidence],-1))
        return logits,{'initial':initial,'states':states,'attention':attention,'final_attention':final_attention,'depth':depth}


def query_indices(queries):return torch.tensor([e*3+h-1 for e,h in queries])


@torch.no_grad()
def metric(logits,labels,mask=None):
    if mask is None:mask=torch.ones_like(labels,dtype=torch.bool)
    if not mask.any():return {'accuracy':None,'ce':None,'queries':0}
    p=logits[mask];y=labels[mask]
    correct=p.argmax(-1)==y
    return {'accuracy':float(correct.float().mean()),
            'macro_accuracy':float(torch.stack([correct[y==c].float().mean() for c in range(4) if (y==c).any()]).mean()),
            'ce':float(F.cross_entropy(p,y)),'queries':len(y)}


@torch.no_grad()
def evaluate(model,source,block):
    q=torch.arange(12)[None].expand(len(source),-1)
    logits,trace=model(source,q);labels=block['labels']
    seen=torch.zeros_like(labels,dtype=torch.bool);seen[:,query_indices(TRAIN_QUERIES)]=True
    result={'seen_queries':metric(logits,labels,seen),'unseen_queries':metric(logits,labels,~seen),
            'all_queries':metric(logits,labels),'by_hops':{},'query_sensitivity':{}}
    for h in [1,2,3]:
        hops=(q%3==h-1)
        result['by_hops'][str(h)]={'seen':metric(logits,labels,seen&hops),'unseen':metric(logits,labels,(~seen)&hops),'all':metric(logits,labels,hops)}
    wrong_start=((q//3+1)%4)*3+q%3
    wrong_hops=q//3*3+(q%3+1)%3
    for name,altered in [('start',wrong_start),('hops',wrong_hops)]:
        alternative,_=model(source,altered);new_labels=labels.gather(1,altered)
        changed=new_labels!=labels
        result['query_sensitivity'][name]={
            'required_answer_change_queries':int(changed.sum()),
            'correct_new_answer':metric(alternative,new_labels,changed),
            'wrong_query_against_original_answer':metric(alternative,labels,changed),
            'prediction_change_on_required_changes':float((alternative.argmax(-1)[changed]!=logits.argmax(-1)[changed]).float().mean()),
            'probability_change_on_required_changes':float((alternative.softmax(-1)-logits.softmax(-1)).abs().amax(-1)[changed].mean())}
    colors=block['facts'][:,4:]
    zero=colors[:,torch.arange(4)].repeat_interleave(3,1)
    one=torch.stack([colors[i,block['facts'][i,:4]] for i in range(len(colors))]).repeat_interleave(3,1)
    difficult=(labels!=zero)&(labels!=one)&(q%3>0)
    result['anti_zero_and_one_hop']=metric(logits,labels,difficult)
    result['shortcut_accuracy']={'source_color':float((zero==labels).float().mean()),'fixed_one_hop':float((one==labels).float().mean()),
                                 'color_mode':float((F.one_hot(colors,4).sum(1).argmax(-1)[:,None]==labels).float().mean())}
    return result


def source_data(data,source):
    train=data['train'][source];center=train.mean(0);scale=train.std(0).clamp_min(1e-6)
    return {split:(block[source]-center)/scale for split,block in data.items() if source in block},center,scale


def train_run(name,kind='mlp',source='oracle',seed=0,steps=8192,lr=.001,small=False,hidden=64,compact=False,depth=1,loop_mode='query'):
    dest=ROOT/'runs'/name;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'summary.json').exists():return json.loads((dest/'summary.json').read_text())
    data=load_data();features,center,scale=source_data(data,source)
    if small:
        data['train']={k:(v[:32] if isinstance(v,torch.Tensor) else v[:32]) for k,v in data['train'].items()}
        features['train']=features['train'][:32]
    torch.manual_seed(seed)
    if kind=='mlp':model=QueryMLP(compact=compact,hidden=hidden,depth=depth)
    elif kind=='recurrent':model=RecurrentQuery(source=source,hidden=hidden,loop_mode=loop_mode)
    else:raise ValueError(kind)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    generator=torch.Generator().manual_seed(42000+seed)
    schedule=torch.randint(len(features['train']),(steps,16),generator=generator)
    save_checkpoint(dest/'schedule.pt',schedule)
    config={'name':name,'kind':kind,'source':source,'seed':seed,'updates':steps,'lr':lr,'small32':small,
            'hidden':hidden,'compact':compact,'depth':depth,'loop_mode':loop_mode,'batch_rules':16,'queries_per_rule':8,
            'parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':'new query network only; cached features frozen'}
    write_json(dest/'config.json',config)
    initial={k:v.clone() for k,v in model.state_dict().items()};curves=[];training=[];start=time.monotonic()
    milestones=sorted(set([s for s in MILESTONES if s<=steps]+[steps]))
    q=query_indices(TRAIN_QUERIES)[None].expand(16,-1)
    def checkpoint(step):
        result={split:evaluate(model,features[split],block) for split,block in data.items() if split in features}
        curves.append({'step':step,'seconds':time.monotonic()-start,'evaluation':result})
        save_checkpoint(dest/f'checkpoint-{step:05d}.pt',{'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                        'step':step,'config':config,'center':center,'scale':scale})
        write_json(dest/'curves.json',curves)
        print(name,step,'train',round(result['train']['seen_queries']['accuracy'],4),
              'val seen',round(result['validation']['seen_queries']['accuracy'],4),
              'val new',round(result['validation']['unseen_queries']['accuracy'],4),flush=True)
    checkpoint(0)
    for step in range(steps):
        indices=schedule[step];logits,_=model(features['train'][indices],q)
        labels=data['train']['labels'][indices][:,query_indices(TRAIN_QUERIES)]
        loss=F.cross_entropy(logits.flatten(0,1),labels.flatten())
        optimizer.zero_grad();loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),1.)
        assert torch.isfinite(loss) and torch.isfinite(norm) and norm>0
        optimizer.step()
        if (step+1)%32==0:training.append({'step':step+1,'loss':float(loss.detach()),'gradient_norm':float(norm)})
        if step+1 in milestones:checkpoint(step+1)
    update=sum((v-initial[k]).square().sum() for k,v in model.state_dict().items()).sqrt().item();assert update>0
    summary={**config,'complete':True,'wall_seconds':time.monotonic()-start,'parameter_update_norm':update,
             'finite_nonzero_gradients':True,'final':curves[-1]['evaluation'],'query_target_draws':steps*16*8}
    write_json(dest/'training.json',training);write_json(dest/'summary.json',summary)
    return summary


def stage1():
    train_run('mlp-original-small32',steps=2048,small=True)
    train_run('mlp-original-seed0')
    train_run('mlp-original-lr0003',lr=.0003)
    train_run('mlp-original-lr003',lr=.003)
    train_run('mlp-capacity128-seed0',hidden=128,compact=True)


def oracle_gate(name='recurrent-oracle-seed0'):
    path=ROOT/'runs'/name/'summary.json'
    s=json.loads(path.read_text())
    values=[s['final'][split][q]['accuracy'] for split in ['validation','balanced'] for q in ['seen_queries','unseen_queries']]
    passed=min(values)>=.9 and s['final']['train']['seen_queries']['accuracy']>=.99
    record={'passed':passed,'minimum_validation_and_balanced_accuracy':min(values),
            'train_seen_accuracy':s['final']['train']['seen_queries']['accuracy'],
            'criterion':'train >=99%; both original validation and balanced diagnostic, seen/unseen queries >=90%',
            'note':'numerical/learning gate only; no reader experiments if oracle composition fails'}
    write_json(ROOT/'oracle-gate.json',record)
    return passed


def stage2_seed0():
    train_run('recurrent-oracle-small32',kind='recurrent',hidden=16,steps=4096,small=True)
    train_run('recurrent-oracle-seed0',kind='recurrent',hidden=16)
    train_run('mlp-matched-seed0',hidden=57,compact=True)
    oracle_gate()


@torch.no_grad()
def prepare_balanced_readers():
    if (ROOT/'balanced-readers.pt').exists():return
    from tokenizers import Tokenizer
    from .pretrained_experiment import model_load
    from .v02_aux import selected_state,rule_text
    from .v02_train import load_run
    model=model_load('cuda');model.requires_grad_(False)
    dream,head,_=load_run(Path('artifacts/wsr-v0.2a/runs/S-seed-0/checkpoint-2048.pt'),'cuda')
    dream.requires_grad_(False);head.requires_grad_(False)
    tokenizer=Tokenizer.from_file('artifacts/pretrained-v1/model/tokenizer.json')
    raw=[];edited=[];rules=balanced_rules()
    for start in range(0,len(rules),8):
        batch=rules[start:start+8]
        tokens=torch.tensor([tokenizer.encode(rule_text(r),add_special_tokens=False).ids for r in batch],device='cuda')
        state=selected_state(model,tokens[:,:-1])
        changed,_=dream(state,model.backbone.embedding(tokens))
        raw.append(head.reader(state).cpu());edited.append(head.reader(changed).cpu())
    save_checkpoint(ROOT/'balanced-readers.pt',{'facts':torch.tensor([list(r[0])+list(r[1]) for r in rules]),
                    'raw_reader':torch.cat(raw),'edited_reader':torch.cat(edited),
                    'model_revision':model.asset_manifest['revision'],'frozen_original_reader_and_Dream':True})


def stage2_more():
    if not oracle_gate():raise RuntimeError('Oracle gate failed; reader expansion disabled')
    for seed in [1,2]:
        train_run(f'mlp-original-seed{seed}',seed=seed)
        train_run(f'mlp-matched-seed{seed}',seed=seed,hidden=57,compact=True)
        train_run(f'recurrent-oracle-seed{seed}',kind='recurrent',seed=seed,hidden=16)
    for mode in ['1','3']:
        train_run('recurrent-fixed'+mode+'-seed0',kind='recurrent',hidden=16,loop_mode=mode)


def stage3():
    if not oracle_gate():raise RuntimeError('Oracle gate failed; do not train reader conditions')
    prepare_balanced_readers();torch.set_num_threads(2)
    for seed in [0,1,2]:
        for source in ['raw_reader','edited_reader']:
            train_run(f'recurrent-{source}-seed{seed}',kind='recurrent',source=source,hidden=16,seed=seed)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['initialize','stage1','stage2-seed0','stage2-more','stage3'])
    args=parser.parse_args();torch.set_num_threads(2);initialize()
    if args.stage=='stage1':stage1()
    elif args.stage=='stage2-seed0':stage2_seed0()
    elif args.stage=='stage2-more':stage2_more()
    elif args.stage=='stage3':stage3()
