"""Eight-read weighted count: visible addresses/weights, final answer supervision only."""
from functools import lru_cache
import hashlib
import itertools
import random
import torch

NODES=16

def bucket(program):
    return int(hashlib.sha256(bytes(program)).hexdigest()[:8],16)%10

def execute(memory,start,program):
    assert int(start)==0
    return sum(int(memory[program[k]][0])*(1+int(program[8+k])) for k in range(8)),list(program[:8])

def family_of(path,program=None):
    if len(set(path))==8:return 'distinct'
    if path[0]==path[4] and len(set(path))==7:return 'reread'
    return None

def world_key(memory):return hashlib.sha256(bytes(row[0] for row in memory)).hexdigest()

@lru_cache(None)
def bit_options(coefficients,target):
    return [bits for bits in itertools.product([0,1],repeat=len(coefficients)) if sum(a*b for a,b in zip(bits,coefficients))==target]

def make_data():
    specs=[('train',4004),('validation_known',260),('validation_new',260),('test_known',260),('test_new',260)]
    used=set();result={}
    for split_no,(split,count) in enumerate(specs):
        rng=random.Random(306000+split_no);records=[]
        for family in ['distinct','reread']:
            for i in range(count):
                label=i%13
                while True:
                    addresses=rng.sample(range(16),8)
                    if family=='reread':addresses[4]=addresses[0]
                    weights=[0]*4+[1]*4;rng.shuffle(weights)
                    program=addresses+weights
                    b=bucket(program)
                    if split in ['train','validation_known','test_known'] and b>=6:continue
                    if split=='validation_new' and not 6<=b<8:continue
                    if split=='test_new' and b<8:continue
                    unique=list(dict.fromkeys(addresses))
                    coefficients=tuple(sum(weights[k]+1 for k,a in enumerate(addresses) if a==address) for address in unique)
                    choices=bit_options(coefficients,label)
                    if not choices:continue
                    bits=rng.choice(choices)
                    memory=[[rng.randrange(2)] for _ in range(16)]
                    for address,bit in zip(unique,bits):memory[address][0]=bit
                    key=world_key(memory)
                    if key in used:continue
                    used.add(key);break
                actual,path=execute(memory,0,program);assert actual==label
                records.append((memory,0,program,label,path,int(family=='reread')))
        rng.shuffle(records)
        result[split]={name:torch.tensor([r[k] for r in records]) for k,name in enumerate(['memory','start','program','label','path','family'])}
    return result

def validate_data(data):
    seen=set();metrics={};queries={}
    for split,block in data.items():
        hist=torch.bincount(block['label'],minlength=13);assert hist.max()==hist.min()
        queries[split]=set()
        for m,s,q,y,path,family in zip(*(block[k].tolist() for k in ['memory','start','program','label','path','family'])):
            answer,reference=execute(m,s,q);assert answer==y and reference==path
            assert family_of(path)==['distinct','reread'][family]
            assert sorted(q[8:])==[0]*4+[1]*4
            key=world_key(m);assert key not in seen;seen.add(key)
            queries[split].add(tuple(q))
        metrics[split]={'examples':len(block['label']),'answer_counts':hist.tolist(),
            'distinct':int((block['family']==0).sum()),'reread':int((block['family']==1).sum()),
            'symbolic_accuracy':1.,'uniform_accuracy':1/13,'mode_baseline_accuracy':1/13,
            'unique_queries':len(queries[split]),'query_buckets':sorted({bucket(q) for q in queries[split]})}
    assert not queries['train']&queries['validation_new']
    assert not queries['train']&queries['test_new']
    assert not queries['validation_new']&queries['test_new']
    return metrics
