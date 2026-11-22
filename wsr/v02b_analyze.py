"""Read-only counterfactuals and post-hoc probes of the final query networks."""
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .v02b_query import ROOT,OLD,QueryMLP,RecurrentQuery,load_data,query_indices,TRAIN_QUERIES,evaluate,metric,oracle_features
from .v02_train import write_json,save_checkpoint
from .v02_small_diagnostics import ridge_predict


def model_load(path):
    saved=torch.load(path,weights_only=True);c=saved['config']
    if c['kind']=='mlp':model=QueryMLP(c['compact'],c['hidden'],c['depth'])
    else:model=RecurrentQuery(c['source'],c['hidden'],c['loop_mode'])
    model.load_state_dict(saved['model']);model.requires_grad_(False)
    return model,saved


def visited(facts,q):
    node=q//3;nodes=[]
    for step in range(3):
        next_node=facts[:,:4].gather(1,node)
        node=torch.where(step<q%3+1,next_node,node);nodes.append(node)
    return nodes


def linear_probe(train,labels,test,test_labels):
    center=train.mean(0);scale=train.std(0).clamp_min(1e-6)
    train=(train-center)/scale;test=(test-center)/scale
    train=F.pad(train,(0,1),value=1).double();test=F.pad(test,(0,1),value=1).double()
    targets=F.one_hot(labels,4).double()
    weights=torch.linalg.solve(train.T@train+.01*torch.eye(train.shape[1]),train.T@targets)
    return {'accuracy':float(((test@weights).argmax(-1)==test_labels).float().mean()),'train_examples':len(train),'validation_examples':len(test)}


@torch.no_grad()
def recurrent_diagnostics(model,saved,data):
    source=saved['config']['source'];center=saved['center'];scale=saved['scale'];result={}
    train=data['train'];train_x=(train[source]-center)/scale
    train_q=torch.arange(12)[None].expand(len(train_x),-1)
    _,train_trace=model(train_x,train_q);train_nodes=visited(train['facts'],train_q)
    train_mask=torch.zeros_like(train_q,dtype=torch.bool);train_mask[:,query_indices(TRAIN_QUERIES)]=True
    for split in ['validation','balanced']:
        block=data[split];x=(block[source]-center)/scale;q=torch.arange(12)[None].expand(len(x),-1)
        logits,trace=model(x,q);nodes=visited(block['facts'],q)
        details={'intact':evaluate(model,x,block),'interventions':{},'forced_depth':{},'posthoc_node_probe':{},'attention_node_accuracy':{}}
        for mode in ['zero','swap_rules','reverse_first_update']:
            changed,other=model(x,q,intervention=mode);active=q%3>=1
            working=torch.zeros_like(trace['states'][0]) if mode=='zero' else trace['states'][0].roll(1,0) if mode=='swap_rules' else 2*trace['initial']-trace['states'][0]
            a=(other['states'][1]-working)[active];b=(trace['states'][1]-trace['states'][0])[active]
            details['interventions'][mode]={'all':metric(changed,block['labels']),
                    'by_hops':{str(h):metric(changed,block['labels'],q%3==h-1) for h in [1,2,3]},
                    'second_state_change_norm':float((other['states'][1]-trace['states'][1])[active].norm(dim=-1).mean()),
                    'second_update_cosine_with_intact':float(F.cosine_similarity(a,b,dim=-1).mean())}
        # Adjacent balanced rules often share a mapping. Add semantic donor controls
        # chosen with oracle analysis labels, without any training or state labels
        # supplied to the query network.
        details['semantic_state_swaps']={}
        for preserving in [True,False]:
            donor=torch.empty_like(q)
            for j in range(q.shape[1]):
                for i in range(len(q)):
                    eligible=(nodes[0][:,j]==nodes[0][i,j]) if preserving else (nodes[0][:,j]!=nodes[0][i,j])
                    eligible[i]=False;choices=eligible.nonzero().flatten()
                    assert len(choices)>0
                    donor[i,j]=choices[(i*13+j*7)%len(choices)]
            replacement=trace['states'][0][donor,torch.arange(12)[None]]
            altered,changed_trace=model(x,q,intervention='replace',replacement=replacement)
            node=nodes[0].gather(0,donor)
            node=torch.where(q%3>0,node,nodes[0])
            for k in range(1,3):node=torch.where(q%3>=k,block['facts'][:,:4].gather(1,node),node)
            counterfactual=block['facts'][:,4:].gather(1,node)
            active=q%3>0
            details['semantic_state_swaps']['preserve_node' if preserving else 'change_node']={
                'against_original_active':metric(altered,block['labels'],active),
                'against_counterfactual_active':metric(altered,counterfactual,active),
                'fraction_counterfactual_answers_differ':float((counterfactual[active]!=block['labels'][active]).float().mean()),
                'donor_first_state_difference_norm':float((replacement-trace['states'][0])[active].norm(dim=-1).mean()),
                'second_state_difference_norm':float((changed_trace['states'][1]-trace['states'][1])[active].norm(dim=-1).mean())}
        for depth in [0,1,2,3]:
            altered,_=model(x,q,forced_rounds=depth)
            actual_node=q//3
            for _ in range(depth):actual_node=block['facts'][:,:4].gather(1,actual_node)
            actual_labels=block['facts'][:,4:].gather(1,actual_node)
            details['forced_depth'][str(depth)]={
                'against_requested_answer':{str(h):metric(altered,block['labels'],q%3==h-1) for h in [1,2,3]},
                'against_actual_forced_depth_answer':metric(altered,actual_labels)}
        for k in range(3):
            fit=train_mask&(train_q%3>=k);test=q%3>=k
            details['posthoc_node_probe'][str(k+1)]=linear_probe(train_trace['states'][k][fit],train_nodes[k][fit],trace['states'][k][test],nodes[k][test])
            expected=q//3 if k==0 else nodes[k-1]
            active=q%3>=k
            details['attention_node_accuracy'][str(k+1)]={'accuracy':float((trace['attention'][k].argmax(-1)[active]==expected[active]).float().mean()),'active_queries':int(active.sum())}
        details['attention_node_accuracy']['final']={'accuracy':float((trace['final_attention'].argmax(-1)==nodes[-1]).float().mean())}
        result[split]=details
        dest=ROOT/'analysis'/saved['config']['name'];dest.mkdir(parents=True,exist_ok=True)
        save_checkpoint(dest/(split+'-states.pt'),{'queries':q,'facts':block['facts'],'labels':block['labels'],
                         'logits':logits,'trace':trace,'note':'post-hoc node labels used only for analysis, never for query-network optimization'})
    return result


@torch.no_grad()
def fact_bridge(data):
    """Privileged localization control; not an answer-only C/D result."""
    previous=json.loads((OLD/'S-results.json').read_text())
    result={'supervision':'extra eight ground-truth primitive facts per training rule; no intermediate node labels; frozen oracle core',
            'limitation':'Uses a supervised fact bridge plus an already trained oracle query core; not matched to final-answer-only C/D and never attributed to Dream',
            'sources':{}}
    for source in ['raw_reader','edited_reader']:
        targets=F.one_hot(data['train']['facts'],4).float().flatten(1)
        predictions=ridge_predict(data['train'][source],targets,[data[s][source] for s in ['validation','balanced']],previous['levels'][source]['penalty'])
        record={}
        for split,p in zip(['validation','balanced'],predictions):
            facts=p.reshape(-1,8,4).argmax(-1)
            values={'fact_accuracy':float((facts==data[split]['facts']).float().mean()),'query_cores':{}}
            for seed in range(3):
                model,saved=model_load(ROOT/'runs'/f'recurrent-oracle-seed{seed}'/'checkpoint-08192.pt')
                x=(oracle_features(facts)-saved['center'])/saved['scale']
                values['query_cores'][str(seed)]=evaluate(model,x,data[split])
            record[split]=values
        result['sources'][source]=record
    write_json(ROOT/'fact-bridge.json',result)


@torch.no_grad()
def main():
    torch.set_num_threads(2);data=load_data();results={}
    for path in sorted((ROOT/'runs').glob('*/summary.json')):
        name=path.parent.name;dest=ROOT/'analysis'/(name+'.json')
        if dest.exists():
            existing=json.loads(dest.read_text())
            if 'recurrent' not in existing or 'semantic_state_swaps' in existing['recurrent']['validation']:
                results[name]=existing;continue
        summary=json.loads(path.read_text());model,saved=model_load(path.parent/f"checkpoint-{summary['updates']:05d}.pt")
        c=saved['config'];source=c['source'];local_data=data
        if c['small32']:
            local_data=dict(data)
            local_data['train']={k:v[:32] for k,v in data['train'].items()}
        result={'config':c,'evaluation':{s:evaluate(model,(b[source]-saved['center'])/saved['scale'],b) for s,b in local_data.items()}}
        if c['kind']=='recurrent' and c['loop_mode']=='query':result['recurrent']=recurrent_diagnostics(model,saved,local_data)
        dest.parent.mkdir(exist_ok=True);write_json(dest,result);results[name]=result
        print('analyzed',name,flush=True)
    fact_bridge(data)
    quality={}
    for split,block in data.items():
        q=torch.arange(12)[None].expand(len(block['facts']),-1);nodes=visited(block['facts'],q)
        symbolic=block['facts'][:,4:].gather(1,nodes[-1]);assert torch.equal(symbolic,block['labels'])
        quality[split]={'rules':len(block['facts']),'symbolic_accuracy':1.,'label_counts':torch.bincount(block['labels'].flatten(),minlength=4).tolist()}
    write_json(ROOT/'data-quality.json',quality)
    write_json(ROOT/'analysis-index.json',{'runs':list(results),'no_intermediate_training_supervision':True})


if __name__=='__main__':main()
