"""Separate LM quality, head learning, state usefulness, iteration benefit, and interventions."""
import argparse
import fcntl
import json
import random
import time

import torch
from torch.nn import functional as F

from .pretrained_experiment import model_load
from .pretrained_evaluate import score,loss_summary
from .state_dream import edit_cache,history_input
from .v02_data import ROOT,LAYER,DISTANCES
from .v02_train import Inputs,load_run,write_json
from .v02_aux import QueryDecoder,FeaturePredictor,TargetBank,TRAIN_QUERIES,HELDOUT_QUERIES,query_tensors,p_key
from .v02_intervene import rollout,trajectory,norm_matched_first
from .v02_train import save_checkpoint


@torch.no_grad()
def final_lm(model,dream,inputs):
    result=[]
    for row in inputs.index['validation']:
        tokens=inputs.tokens[row['token_path']];state=inputs.load(row);b=row['boundary']
        edited,trace,initial=edit_cache(model,dream,state,tokens,b,128,LAYER)
        values,_=score(model,tokens,edited,b,min(2048,tokens.shape[1]-b))
        result.append({'id':row['id'],'document':row['document'],'boundary':b,'losses':values,
                       'history_visible':b,'state_history':state.processed,'history_window':128,'selected_layer':LAYER,
                       **loss_summary(values),'first_64_nll':sum(values[:64])/64,
                       'trajectory':trajectory(initial,trace)})
    return result


@torch.no_grad()
def panel(model,dream,inputs):
    results=[]
    for row in inputs.curve:
        tokens=inputs.tokens[row['token_path']];state=inputs.load(row);b=row['boundary']
        initial=state.layers[LAYER].ssm;history=history_input(model,tokens,b,128)
        final,trace,memories=rollout(dream,initial,history)
        normal,_=dream(initial,history)
        torch.testing.assert_close(final,normal,rtol=0,atol=0)
        readings={}
        for r in [0,1,2,4,8]:
            h=initial if r==0 else trace[r-1]
            values,_=score(model,tokens,state.edit(LAYER,h),b,2048)
            readings[str(r)]={'losses':values,**loss_summary(values)}
        interventions={}
        matched=norm_matched_first(initial,trace)
        matched_values,_=score(model,tokens,state.edit(LAYER,matched),b,2048)
        for mode in ['reset_state','reset_memory','reverse_state']:
            h,changed,z=rollout(dream,initial,history,mode)
            values,_=score(model,tokens,state.edit(LAYER,h),b,512)
            after=initial if mode=='reset_state' else 2*initial-trace[3] if mode=='reverse_state' else trace[3]
            d=(changed[4]-after).double().flatten();base=(trace[4]-trace[3]).double().flatten()
            interventions[mode]={**loss_summary(values),
                'final_change_vs_intact_relative':float((h-final).double().norm()/initial.double().norm()),
                'first_post_intervention_delta_cosine':float(F.cosine_similarity(d[None],base[None])),
                'first_post_intervention_delta_norm_ratio':float(d.norm()/base.norm()),
                'final_memory_change':float((z[-1]-memories[-1]).norm())}
        results.append({'id':row['id'],'history_visible':b,'state_history':state.processed,
                        'rounds':readings,'interventions_after_round':4,'interventions':interventions,
                        'norm_matched_first_edit':{'losses':matched_values,**loss_summary(matched_values),
                            'cosine_final_with_first_total':float(F.cosine_similarity((trace[0]-initial).double().flatten()[None],(trace[-1]-initial).double().flatten()[None]))},
                        'trajectory':trajectory(initial,trace),
                        'note':'Readouts from the same model trained for 8 rounds; interventions reuse the original learned GRU and editor.'})
    return results


@torch.no_grad()
def periodic(model,dream,inputs,total=32768):
    results=[]
    for row in inputs.index['validation']:
        if row['document']['dataset']!='pg19' or row['boundary']!=512:continue
        tokens=inputs.tokens[row['token_path']];state=inputs.load(row);boundary=512;events=[];losses=[]
        while boundary<total:
            edited,trace,initial=edit_cache(model,dream,state,tokens,boundary,128,LAYER)
            length=min(2048,total-boundary)
            values,state=score(model,tokens,edited,boundary,length)
            events.append({'history':boundary,'state_history_before':boundary-1,'processed_after':state.processed,'nll':sum(values)/len(values),
                           'losses':values,'relative_edit':0. if not trace else float((trace[-1]-initial).norm()/initial.norm())})
            losses+=values;boundary+=length
        results.append({'document':row['document'],'events':events,'tokens':len(losses),'mean_nll':sum(losses)/len(losses),
                        'resets_within_book':0,'total_history':total})
    return results


@torch.no_grad()
def save_representative_trajectories(model,dream,inputs,path):
    rows=[]
    for row in [inputs.curve[0],inputs.curve[-1]]:
        state=inputs.load(row);tokens=inputs.tokens[row['token_path']];initial=state.layers[LAYER].ssm
        history=history_input(model,tokens,row['boundary'],128)
        final,trace,memories=rollout(dream,initial,history)
        controls={mode:rollout(dream,initial,history,mode)[0].cpu() for mode in ['reset_state','reset_memory','reverse_state']}
        rows.append({'id':row['id'],'document':row['document'],'boundary':row['boundary'],
            'initial_ssm':initial.cpu(),'round_states':[h.cpu() for h in trace],
            'round_GRU_memories':[z.cpu() for z in memories],'counterfactual_final_states':controls})
    save_checkpoint(path,rows)


def class_metrics(logits,labels):
    return {'correct':int((logits.argmax(-1)==labels).sum()),'count':labels.numel(),
            'ce_sum':float(F.cross_entropy(logits.flatten(0,1),labels.flatten(),reduction='sum'))}


def merge_metrics(rows):
    count=sum(r['count'] for r in rows)
    return {'accuracy':sum(r['correct'] for r in rows)/count,'ce':sum(r['ce_sum'] for r in rows)/count,'queries':count}


@torch.no_grad()
def s_evaluate(model,dream,head,control,seed,step=2048):
    validation=torch.load(ROOT/'aux-s/validation.pt',weights_only=True,mmap=True)
    training=torch.load(ROOT/'aux-s/train.pt',weights_only=True,mmap=True)
    seen=sorted({random.Random(seed*100000+i).randrange(len(training)) for i in range(step)})[:128]
    pools=[('new_rules_seen_queries',validation,TRAIN_QUERIES),
           ('new_rules_new_queries',validation,HELDOUT_QUERIES),
           ('seen_rules_new_queries',[training[i] for i in seen],HELDOUT_QUERIES)]
    if step<2048:pools=pools[:2];pools=[(name,rows[:64],queries) for name,rows,queries in pools]
    result={}
    device=next(dream.parameters()).device
    for name,examples,queries in pools:
        all_metrics={str(r):[] for r in [0,1,2,4,8]}
        for condition in ['no_Dream_trained_head','norm_matched_first_edit','shuffled_queries_8','visible_color_frequency_baseline']:all_metrics[condition]=[]
        for example in examples:
            initial=example['state'].to(device);tokens=example['tokens'].to(device)
            final,trace=dream(initial,history_input(model,tokens,tokens.shape[1],128))
            q,labels=query_tensors(example['rule'],queries,device)
            for r in [0,1,2,4,8]:
                h=initial if r==0 else trace[r-1]
                all_metrics[str(r)].append(class_metrics(head(h,q),labels))
            all_metrics['no_Dream_trained_head'].append(class_metrics(control(initial,q),labels))
            all_metrics['norm_matched_first_edit'].append(class_metrics(head(norm_matched_first(initial,trace),q),labels))
            all_metrics['shuffled_queries_8'].append(class_metrics(head(final,q.roll(1,1)),labels))
            counts=torch.bincount(torch.tensor(example['rule'][1],device=device),minlength=4).float()+.25
            frequency_logits=counts.log()[None,None].expand(1,q.shape[1],-1)
            all_metrics['visible_color_frequency_baseline'].append(class_metrics(frequency_logits,labels))
        result[name]={k:merge_metrics(v) for k,v in all_metrics.items()}
    result['random_accuracy']=.25
    result['description']='0: jointly trained query head on raw state; no_Dream_trained_head: separately trained head, same initialization/instances/queries/updates, raw state only'
    return result


@torch.no_grad()
def p_evaluate(model,dream,head,control,inputs,step=2048):
    bank=TargetBank('validation',inputs.device);results=[];predicted={str(r):[[],[],[]] for r in [0,1,2,4,8]}
    predicted['no_Dream_trained_head']=[[],[],[]]
    train_bank=TargetBank('train',inputs.device);priors=[]
    for distance in DISTANCES:
        per_dataset=[]
        for dataset in ['wikitext103','pg19']:
            indices=[train_bank.lookup[p_key(row,cycle,distance)] for row in inputs.index['train']
                     if row['document']['dataset']==dataset for cycle in range(2)]
            per_dataset.append(train_bank.features[indices].mean(0))
        priors.append(F.normalize(torch.stack(per_dataset).mean(0),dim=-1))
    priors=torch.stack(priors)[None]
    rows=inputs.index['validation'] if step==2048 else inputs.curve
    for row in rows:
        if not all(p_key(row,0,d) in bank.lookup for d in DISTANCES):continue
        state=inputs.load(row);initial=state.layers[LAYER].ssm;tokens=inputs.tokens[row['token_path']]
        final,trace=dream(initial,history_input(model,tokens,row['boundary'],128))
        per_mode={}
        for within in [True,False]:
            candidates,labels=bank.targets(row,0,random.Random(913+row['boundary']),within)
            metrics={}
            for r in [0,1,2,4,8]:
                h=initial if r==0 else trace[r-1]
                prediction=head(h,torch.tensor([[0,1,2]],device=inputs.device))
                if within:
                    for j in range(3):predicted[str(r)][j].append(prediction[0,j].cpu())
                logits=torch.einsum('bdh,bdkh->bdk',prediction,candidates)/.1
                metrics[str(r)]={'per_distance':[{'distance':d,'correct':int(logits[0,j].argmax()==labels[0,j]),
                               'ce':float(F.cross_entropy(logits[:,j],labels[:,j]))} for j,d in enumerate(DISTANCES)]}
            control_prediction=control(initial,torch.tensor([[0,1,2]],device=inputs.device))
            if within:
                for j in range(3):predicted['no_Dream_trained_head'][j].append(control_prediction[0,j].cpu())
            logits=torch.einsum('bdh,bdkh->bdk',control_prediction,candidates)/.1
            metrics['no_Dream_trained_head']={'per_distance':[{'distance':d,'correct':int(logits[0,j].argmax()==labels[0,j]),
                            'ce':float(F.cross_entropy(logits[:,j],labels[:,j]))} for j,d in enumerate(DISTANCES)]}
            prediction=head(norm_matched_first(initial,trace),torch.tensor([[0,1,2]],device=inputs.device))
            logits=torch.einsum('bdh,bdkh->bdk',prediction,candidates)/.1
            metrics['norm_matched_first_edit']={'per_distance':[{'distance':d,'correct':int(logits[0,j].argmax()==labels[0,j]),
                            'ce':float(F.cross_entropy(logits[:,j],labels[:,j]))} for j,d in enumerate(DISTANCES)]}
            for name,prediction in [('distance_only_train_prior',priors),('candidate_centrality',F.normalize(candidates.mean(2),dim=-1))]:
                logits=torch.einsum('bdh,bdkh->bdk',prediction,candidates)/.1
                metrics[name]={'per_distance':[{'distance':d,'correct':int(logits[0,j].argmax()==labels[0,j]),
                                'ce':float(F.cross_entropy(logits[:,j],labels[:,j]))} for j,d in enumerate(DISTANCES)]}
            prediction=head(final,torch.tensor([[0,1,2]],device=inputs.device))
            shuffled=torch.einsum('bdh,bdkh->bdk',prediction,candidates.roll(1,2))/.1
            metrics['permuted_targets']={'per_distance':[{'distance':d,'correct':int(shuffled[0,j].argmax()==labels[0,j]),
                        'ce':float(F.cross_entropy(shuffled[:,j],labels[:,j]))} for j,d in enumerate(DISTANCES)]}
            per_mode['within_document' if within else 'cross_document']=metrics
        results.append({'id':row['id'],'metrics':per_mode})
    collapse={}
    for condition,distances in predicted.items():
        collapse[condition]={}
        for distance,vectors in zip(DISTANCES,distances):
            vectors=torch.stack(vectors).double();centered=vectors-vectors.mean(0)
            singular=torch.linalg.svdvals(centered);energy=singular.square();prob=energy/energy.sum().clamp_min(1e-20)
            collapse[condition][str(distance)]={'states':len(vectors),'coordinate_std_mean':float(vectors.std(0).mean()),
                'effective_rank':float((-prob[prob>0]*prob[prob>0].log()).sum().exp()),
                'mean_pairwise_cosine':float((vectors.sum(0).square().sum()-len(vectors))/(len(vectors)*(len(vectors)-1)))}
    return {'rows':results,'random_accuracy':.125,'prediction_collapse_diagnostics':collapse,
            'description':'same-document hard negatives, non-overlapping; cross-document negatives and target permutation are shortcut checks, not training objectives'}


@torch.no_grad()
def feature_diagnostics():
    result={}
    for split in ['train','validation']:
        raw=torch.load(ROOT/'aux-p'/(split+'-raw.pt'),weights_only=True)
        data=torch.load(ROOT/'aux-p'/(split+'.pt'),weights_only=True)
        features=data['features'].float();sample=features[torch.linspace(0,len(features)-1,min(2048,len(features))).long()]
        singular=torch.linalg.svdvals(sample.double()-sample.double().mean(0))
        energy=singular.square();prob=energy/energy.sum()
        rng=random.Random(9);within=[];cross=[]
        bank=TargetBank(split,'cpu')
        for i in rng.sample(range(len(features)),min(512,len(features))):
            key=bank.keys[i];same=[j for j in bank.documents[key[0]] if abs(bank.keys[j][1]-key[1])>=64]
            other=[j for j,k in enumerate(bank.keys) if k[0]!=key[0]]
            if same:within.append(float(features[i]@features[rng.choice(same)]))
            cross.append(float(features[i]@features[rng.choice(other)]))
        result[split]={'features':len(features),'dimension':features.shape[1],
            'norm_min':float(features.norm(dim=-1).min()),'norm_max':float(features.norm(dim=-1).max()),
            'raw_coordinate_std_mean':float(raw['features'].std(0).mean()),
            'normalized_coordinate_std_mean':float(features.std(0).mean()),
            'effective_rank_entropy':float((-prob[prob>0]*prob[prob>0].log()).sum().exp()),
            'top_principal_energy_fraction':float(prob[0]),'mean_same_document_cosine':sum(within)/len(within),
            'mean_cross_document_cosine':sum(cross)/len(cross),'all_finite':bool(torch.isfinite(features).all())}
        index=json.loads((ROOT/'bank/index.json').read_text())[split]
        maximum_cosines=[];oracle=[]
        for row in index:
            if not all(p_key(row,0,d) in bank.lookup for d in DISTANCES):continue
            candidates,labels=bank.targets(row,0,random.Random(913+row['boundary']))
            for j,distance in enumerate(DISTANCES):
                positive=bank.features[bank.lookup[p_key(row,0,distance)]]
                similarities=candidates[0,j]@positive
                oracle.append(int(similarities.argmax()==labels[0,j]))
                mask=torch.arange(8)!=labels[0,j]
                maximum_cosines.append(float(similarities[mask].max()))
        result[split].update({'target_self_match_accuracy':sum(oracle)/len(oracle),
            'candidate_sets_checked':len(oracle),'mean_max_positive_negative_cosine':sum(maximum_cosines)/len(maximum_cosines),
            'near_duplicate_negative_fraction_above_0999':sum(v>.999 for v in maximum_cosines)/len(maximum_cosines)})
    return result


def _evaluate_auxiliary(group,seed,model,inputs):
    directory=ROOT/'evaluation'/f'{group}-seed-{seed}';directory.mkdir(parents=True,exist_ok=True)
    for step in [512,1024,2048]:
        dest=directory/f'auxiliary-{step}.json'
        if dest.exists():continue
        path=ROOT/'runs'/f'{group}-seed-{seed}'/f'checkpoint-{step:04d}.pt'
        dream,head,saved=load_run(path,inputs.device)
        control=(QueryDecoder() if group=='S' else FeaturePredictor()).to(inputs.device)
        control.load_state_dict(saved['control_head'])
        result=s_evaluate(model,dream,head,control,seed,step) if group=='S' else p_evaluate(model,dream,head,control,inputs,step)
        result['evaluation_device']=str(inputs.device)
        write_json(dest,result);print('evaluated',group,seed,'auxiliary',step,str(inputs.device),flush=True)


def evaluate_auxiliary(group,seed,model,inputs):
    if group=='S' and (ROOT/'scope-update.json').exists():
        print('S auxiliary evaluation paused; existing results retained',seed,flush=True);return
    directory=ROOT/'evaluation'/f'{group}-seed-{seed}';directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.aux-evaluation.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _evaluate_auxiliary(group,seed,model,inputs)


def evaluate(group,seed,model,inputs):
    scope=ROOT/'scope-update.json'
    if scope.exists() and group!='baseline' and group not in json.loads(scope.read_text())['active_expanded_evaluation_groups']:
        print('expanded evaluation paused by scope update:',group,seed,flush=True);return
    directory=ROOT/'evaluation'/f'{group}-seed-{seed}';directory.mkdir(parents=True,exist_ok=True)
    if group=='baseline':
        if not (directory/'lm.json').exists():write_json(directory/'lm.json',final_lm(model,None,inputs))
        if not (directory/'periodic.json').exists():write_json(directory/'periodic.json',periodic(model,None,inputs))
        return
    path=ROOT/'runs'/f'{group}-seed-{seed}'/'checkpoint-2048.pt'
    dream,head,saved=load_run(path,inputs.device)
    if not (directory/'state-trajectories.pt').exists():
        save_representative_trajectories(model,dream,inputs,directory/'state-trajectories.pt')
    for name,function in [('lm',lambda:final_lm(model,dream,inputs)),('panel',lambda:panel(model,dream,inputs))]:
        if not (directory/(name+'.json')).exists():
            write_json(directory/(name+'.json'),function());print('evaluated',group,seed,name,flush=True)
    if seed==0 and not (directory/'periodic.json').exists():
        write_json(directory/'periodic.json',periodic(model,dream,inputs));print('evaluated',group,seed,'periodic',flush=True)
    if group in ['S','P']:
        if not all((directory/f'auxiliary-{step}.json').exists() for step in [512,1024,2048]):
            # Uniform CPU FP32 auxiliary readout; avoids competing with GPU training.
            auxiliary_model=model_load('cpu');auxiliary_inputs=Inputs(auxiliary_model)
            evaluate_auxiliary(group,seed,auxiliary_model,auxiliary_inputs)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--groups',nargs='+',default=['baseline','B0','P'])
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);p.add_argument('--features-only',action='store_true')
    p.add_argument('--auxiliary-only',action='store_true');p.add_argument('--device',choices=['cpu','cuda'])
    a=p.parse_args()
    if a.features_only:
        torch.set_num_threads(4);write_json(ROOT/'feature-diagnostics.json',feature_diagnostics())
    else:
        device=a.device or ('cpu' if a.auxiliary_only else 'cuda')
        model=model_load(device);inputs=Inputs(model)
        for group in a.groups:
            for seed in ([0] if group=='baseline' else a.seeds):
                if a.auxiliary_only:
                    if group not in ['S','P']:raise ValueError('Auxiliary evaluation is only defined for S/P')
                    evaluate_auxiliary(group,seed,model,inputs)
                else:evaluate(group,seed,model,inputs)
