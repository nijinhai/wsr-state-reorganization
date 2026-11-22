"""Auxiliary signals: composed rule queries and hard-negative future features.

Neither decoder receives history. Future target features are never passed to Dream.
"""
import argparse
import itertools
import json
import random

import torch
from torch import nn
from torch.nn import functional as F

from .v02_data import ROOT,LAYER,DISTANCES
from .pretrained_experiment import model_load


COLORS=('red','green','blue','gold')
NAMES=('A','B','C','D')
ALL_QUERIES=tuple(itertools.product(range(4),range(1,4)))
TRAIN_QUERIES=tuple(q for q in ALL_QUERIES if (q[0]+q[1])%3!=0)
HELDOUT_QUERIES=tuple(q for q in ALL_QUERIES if q not in TRAIN_QUERIES)


def answer(rule,query):
    mapping,colors=rule;entity,hops=query
    for _ in range(hops):entity=mapping[entity]
    return colors[entity]


def rule_text(rule):
    mapping,colors=rule
    lines=['There are four locations: A, B, C, and D. Each has a color and a directed link.']
    lines += [f'Location {NAMES[i]} has color {COLORS[colors[i]]}. Its link leads to {NAMES[mapping[i]]}.' for i in range(4)]
    lines += ['A walk follows the directed links. The requested color is the color at the final location.']
    return '\n'.join(lines)


def generate_rules():
    rules=list(itertools.product(itertools.permutations(range(4)),itertools.product(range(4),repeat=4)))
    random.Random(20261116).shuffle(rules)
    return {'train':rules[:1024],'validation':rules[1024:1280]}


def s_data_quality():
    from collections import Counter
    out={}
    for split,rules in generate_rules().items():
        out[split]={}
        for name,queries in [('trained_combinations',TRAIN_QUERIES),('heldout_combinations',HELDOUT_QUERIES)]:
            labels=[answer(rule,q) for rule in rules for q in queries]
            mode=[max(range(4),key=lambda c:rule[1].count(c)) for rule in rules for q in queries]
            zero=[rule[1][q[0]] for rule in rules for q in queries]
            one=[rule[1][rule[0][q[0]]] for rule in rules for q in queries]
            accuracy=lambda values:sum(a==b for a,b in zip(labels,values))/len(labels)
            out[split][name]={'samples':len(rules),'queries':len(labels),'label_frequencies':dict(Counter(labels)),
                'uniform_random_accuracy':.25,'visible_color_mode_accuracy':accuracy(mode),
                'zero_hop_lookup_accuracy':accuracy(zero),'one_hop_lookup_ignoring_requested_hops_accuracy':accuracy(one)}
    lengths=[]
    for split in ['train','validation']:
        lengths.extend(row['tokens'].numel() for row in torch.load(ROOT/'aux-s'/(split+'.pt'),weights_only=True,mmap=True))
    out.update({'history_tokens_min':min(lengths),'history_tokens_max':max(lengths),'history_window':128,
                'shortcut_scope':'data-level oracles given visible structured facts; no learned state readout, not comparable model capacity'})
    (ROOT/'S-data-quality.json').write_text(json.dumps(out,indent=2))
    return out


class StateReader(nn.Module):
    """Small shared readout: state only, no token/history/target access."""
    def __init__(self):
        super().__init__();self.row=nn.Linear(128,16);self.project=nn.Linear(24*17,128)

    def forward(self,state):
        scale=state.square().mean((2,3),keepdim=True).sqrt().clamp_min(1e-4)
        rows=F.silu(self.row(state/scale)).mean(2)
        values=torch.cat([rows,scale.flatten(2).log()],-1).flatten(1)
        return F.silu(self.project(values))


class QueryDecoder(nn.Module):
    def __init__(self):
        super().__init__();self.reader=StateReader();self.entity=nn.Embedding(4,8);self.hops=nn.Embedding(3,8)
        self.output=nn.Sequential(nn.Linear(144,64),nn.SiLU(),nn.Linear(64,4))

    def forward(self,state,queries):
        # Queries arrive after the single shared state edit. No query-specific Dream.
        z=self.reader(state)[:,None].expand(-1,queries.shape[1],-1)
        query=torch.cat([self.entity(queries//3),self.hops(queries%3)],-1)
        return self.output(torch.cat([z,query],-1))


class FeaturePredictor(nn.Module):
    def __init__(self):
        super().__init__();self.reader=StateReader();self.distance=nn.Embedding(3,16)
        self.output=nn.Sequential(nn.Linear(144,128),nn.SiLU(),nn.Linear(128,768))

    def forward(self,state,distances):
        z=self.reader(state)[:,None].expand(-1,distances.shape[1],-1)
        return F.normalize(self.output(torch.cat([z,self.distance(distances)],-1)),dim=-1)


def query_tensors(rule,queries,device):
    q=torch.tensor([[e*3+h-1 for e,h in queries]],device=device)
    a=torch.tensor([[answer(rule,t) for t in queries]],device=device)
    return q,a


@torch.no_grad()
def selected_state(model,tokens):
    """Exact first 13 official layers; upper layers cannot affect the selected cache."""
    x=model.backbone.embedding(tokens);residual=None;selected=None
    for layer in model.backbone.layers[:LAYER+1]:
        residual=x if residual is None else residual+x
        x=layer.norm(residual)
        x,selected=layer.mixer(x,layer.mixer.initial(tokens.shape[0],tokens.device,x.dtype))
    return selected.ssm


@torch.no_grad()
def prepare_s(device='cuda'):
    from tokenizers import Tokenizer
    model=model_load(device);tokenizer=Tokenizer.from_file('artifacts/pretrained-v1/model/tokenizer.json')
    directory=ROOT/'aux-s';directory.mkdir(parents=True,exist_ok=True)
    rules=generate_rules()
    for split,instances in rules.items():
        dest=directory/(split+'.pt')
        if dest.exists():continue
        rows=[]
        for i,rule in enumerate(instances):
            tokens=torch.tensor(tokenizer.encode(rule_text(rule),add_special_tokens=False).ids,device=device)[None]
            state=selected_state(model,tokens[:,:-1])
            rows.append({'rule':rule,'state':state.cpu(),'tokens':tokens.cpu(),'history':rule_text(rule)})
            if (i+1)%64==0:print('S bank',split,i+1,flush=True)
        torch.save(rows,dest)
    example=rules['validation'][0]
    validation={name:[sum(answer(r,q)==c for r in instances for q in ALL_QUERIES)/(len(instances)*12) for c in range(4)] for name,instances in rules.items()}
    (directory/'protocol.json').write_text(json.dumps({'train_rule_instances':1024,'validation_rule_instances':256,
        'total_rule_space':6144,'train_queries':TRAIN_QUERIES,'unseen_query_combinations':HELDOUT_QUERIES,
        'query_example':{'history':rule_text(example),'query':'Starting at A, follow 2 links. What is the final color?',
                         'answer':COLORS[answer(example,(0,2))]},
        'answers':COLORS,'random_accuracy':0.25,'label_frequencies':validation,
        'decoder_input':'shared edited SSM state + query ID only; no history',
        'Dream_input':'state and last 128 visible history embeddings; no queries or derived labels',
        'heldout_evaluations':['new_rules_seen_queries','seen_rules_new_queries','new_rules_new_queries'],
        'rule_identity_split_disjoint':not bool(set(rules['train'])&set(rules['validation']))},indent=2))
    s_data_quality()


def p_key(row,cycle,distance):
    return row['document']['dataset']+'-'+row['document']['id'],row['boundary']+256*cycle+distance-1


@torch.no_grad()
def prepare_p(device='cuda'):
    model=model_load(device);bank=json.loads((ROOT/'bank/index.json').read_text())
    directory=ROOT/'aux-p';directory.mkdir(parents=True,exist_ok=True)
    rawfiles=[]
    for split in ['train','validation']:
        rawfile=directory/(split+'-raw.pt');rawfiles.append(rawfile)
        if rawfile.exists():continue
        windows={}
        for row in bank[split]:
            tokens=torch.load(row['token_path'],weights_only=True)['tokens']
            for cycle in range(2):
                for distance in DISTANCES:
                    key=p_key(row,cycle,distance)
                    if key[1]+64<=len(tokens):windows[key]=tokens[key[1]:key[1]+64]
            doc=p_key(row,0,128)[0]
            for pos in torch.linspace(0,len(tokens)-64,16).long().tolist():
                windows[(doc,pos)]=tokens[pos:pos+64]
        keys=sorted(windows);features=[]
        for start in range(0,len(keys),8):
            batch=torch.stack([windows[k] for k in keys[start:start+8]]).to(device)
            hidden,_=model.run(batch,return_hidden=True)
            features.append(hidden.mean(1).cpu())
            if start%512==0:print('P targets',split,start,'/',len(keys),flush=True)
        torch.save({'keys':keys,'features':torch.cat(features)},rawfile)
    train=torch.load(rawfiles[0],weights_only=True)
    center=train['features'].mean(0);scale=train['features'].std(0).clamp_min(.05)
    for split,path in zip(['train','validation'],rawfiles):
        raw=torch.load(path,weights_only=True)
        torch.save({'keys':raw['keys'],'features':F.normalize((raw['features']-center)/scale,dim=-1),
                    'train_center':center,'train_scale':scale},directory/(split+'.pt'))
    (directory/'protocol.json').write_text(json.dumps({'encoder':'frozen official Mamba2-130M final normalized hidden, independent 64-token segment, mean pooling',
        'feature_dimension':768,'normalization':'train-only per-coordinate centering/scaling then L2',
        'distances':DISTANCES,'distance_definition':'1 means the first future token at index boundary',
        'candidates':8,'negative_control':'same document, non-overlapping 64-token windows; candidates permuted; cross-document comparison reported',
        'temperature':0.1,'shortcut_checks':['within-document vs cross-document negatives','target-vector permutation control','unedited state readout'],
        'frozen_encoder_gradients':False,'future_features_given_to_Dream':False},indent=2))


class TargetBank:
    def __init__(self,split,device):
        data=torch.load(ROOT/'aux-p'/(split+'.pt'),weights_only=True)
        self.keys=[tuple(k) for k in data['keys']];self.lookup={k:i for i,k in enumerate(self.keys)}
        self.features=data['features'].to(device);self.documents={}
        for i,(doc,pos) in enumerate(self.keys):self.documents.setdefault(doc,[]).append(i)

    def candidates(self,key,rng,within=True,count=8):
        positive=self.lookup[key]
        pool=self.documents[key[0]] if within else range(len(self.keys))
        negative=[i for i in pool if (self.keys[i][0]!=key[0] if not within else abs(self.keys[i][1]-key[1])>=64)]
        rng.shuffle(negative);chosen=[positive]
        for candidate in negative:
            doc,pos=self.keys[candidate]
            if all(self.keys[j][0]!=doc or abs(self.keys[j][1]-pos)>=64 for j in chosen):chosen.append(candidate)
            if len(chosen)==count:break
        if len(chosen)<count:raise ValueError('Insufficient non-overlapping hard negatives')
        rng.shuffle(chosen)
        return self.features[chosen],chosen.index(positive)

    def targets(self,row,cycle,rng,within=True):
        values=[];labels=[]
        for distance in DISTANCES:
            value,label=self.candidates(p_key(row,cycle,distance),rng,within)
            values.append(value);labels.append(label)
        return torch.stack(values)[None],torch.tensor([labels],device=self.features.device)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('signal',choices=['S','P']);p.add_argument('--device',default='cuda')
    a=p.parse_args();prepare_s(a.device) if a.signal=='S' else prepare_p(a.device)
