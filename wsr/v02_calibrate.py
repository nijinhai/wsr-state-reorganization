"""Choose S/P weights separately, using training-only gradient scales and short trials."""
import copy
import json
import random
import statistics

import torch
from torch.nn import functional as F

from .state_dream import StateDream,edit_cache,history_input
from .pretrained_experiment import model_load,continuation
from .v02_data import ROOT,LAYER
from .v02_aux import QueryDecoder,FeaturePredictor,TargetBank,TRAIN_QUERIES,query_tensors
from .v02_train import Inputs,schedule,gradient_stats,write_json


def objectives(model,dream,head,inputs,row,step,group,s_data,p_data):
    device=inputs.device;tokens=inputs.tokens[row['token_path']];state=inputs.load(row);boundary=row['boundary']
    lm=[];aux=[]
    for cycle in range(2):
        edited,_,_=edit_cache(model,dream,state,tokens,boundary,128,LAYER)
        loss,advanced=continuation(model,tokens,edited,boundary,64);lm.append(loss)
        if group=='P':
            candidates,labels=p_data.targets(row,cycle,random.Random(step*10+cycle))
            prediction=head(edited.layers[LAYER].ssm,torch.tensor([[0,1,2]],device=device))
            logits=torch.einsum('bdh,bdkh->bdk',prediction,candidates)/.1
            aux.append(F.cross_entropy(logits.flatten(0,1),labels.flatten()))
        state=advanced.detach()
        if cycle==0:
            boundary+=256;state=model.wake(tokens[:,state.processed:boundary-1],state)
    if group=='S':
        example=s_data[random.Random(step).randrange(len(s_data))];visible=example['tokens'].to(device)
        raw=example['state'].to(device)
        shared,_=dream(raw,history_input(model,visible,visible.shape[1],128))
        queries,labels=query_tensors(example['rule'],TRAIN_QUERIES,device)
        aux.append(F.cross_entropy(head(shared,queries).flatten(0,1),labels.flatten()))
    return torch.stack(lm).mean(),torch.stack(aux).mean()


def main():
    path=ROOT/'calibration.json'
    if path.exists():print('Preserved calibration');return
    model=model_load('cuda');inputs=Inputs(model);order=schedule(inputs.index['train'],0)
    s_data=torch.load(ROOT/'aux-s/train.pt',weights_only=True,mmap=True);p_data=TargetBank('train','cuda')
    result={'policy':'separate median initial Dream-gradient norm matching; desired weighted auxiliary/LM norm=0.3; train-only 8-update stability trials at 0.5x and 1x; choose 1x if finite and median weighted gradient ratio <=1, otherwise 0.5x; if needed reduce to ratio 0.3 using the smoke measurement',
            'validation_used_for_weights':False,'measurement_samples':16,'weights':{},'groups':{}}
    for group in ['S','P']:
        torch.manual_seed(0);dream=StateDream(mode='D').cuda()
        head=(QueryDecoder() if group=='S' else FeaturePredictor()).cuda()
        measurements=[]
        for step in range(16):
            lm,aux=objectives(model,dream,head,inputs,order[step],step,group,s_data,p_data)
            stats=gradient_stats(lm,aux,list(dream.parameters()));measurements.append(stats)
            print('calibrate',group,step,stats,flush=True)
        median_lm=statistics.median(r['LM'] for r in measurements)
        median_aux=statistics.median(r['auxiliary_unweighted'] for r in measurements)
        if not median_lm>0 or not median_aux>0:raise RuntimeError('One training signal has no Dream gradient')
        proposed=min(10.,max(1e-5,.3*median_lm/max(median_aux,1e-20)))
        original_dream=copy.deepcopy(dream.state_dict());original_head=copy.deepcopy(head.state_dict())
        trials=[]
        for multiplier in [.5,1.]:
            dream.load_state_dict(original_dream);head.load_state_dict(original_head)
            optimizer=torch.optim.Adam(list(dream.parameters())+list(head.parameters()),lr=3e-4)
            values=[]
            for step in range(8):
                optimizer.zero_grad(set_to_none=True)
                lm,aux=objectives(model,dream,head,inputs,order[step],step,group,s_data,p_data)
                components=gradient_stats(lm,aux,list(dream.parameters()))
                loss=lm+proposed*multiplier*aux
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(dream.parameters(),1.,error_if_nonfinite=True)
                torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                values.append({'LM':float(lm.detach()),'auxiliary':float(aux.detach()),'Dream_gradient_norm':float(norm),
                               'weighted_auxiliary_LM_gradient_ratio':proposed*multiplier*components['auxiliary_unweighted']/max(components['LM'],1e-20)})
            finite=all(all(torch.isfinite(torch.tensor(v)) for v in r.values()) and r['Dream_gradient_norm']>0 for r in values)
            trials.append({'multiplier':multiplier,'weight':proposed*multiplier,'finite_nonzero':finite,
                           'median_weighted_gradient_ratio':statistics.median(r['weighted_auxiliary_LM_gradient_ratio'] for r in values),'steps':values})
        chosen=trials[1] if trials[1]['finite_nonzero'] and trials[1]['median_weighted_gradient_ratio']<=1. else trials[0]
        if not chosen['finite_nonzero']:raise RuntimeError('Auxiliary signal failed calibration trials')
        final_weight=chosen['weight']
        if chosen['median_weighted_gradient_ratio']>1.:final_weight*=.3/chosen['median_weighted_gradient_ratio']
        result['weights'][group]=final_weight
        result['groups'][group]={'initial_gradients':measurements,'median_LM':median_lm,'median_auxiliary':median_aux,
                                 'initial_weighted_gradient_ratio':final_weight*median_aux/median_lm,
                                 'chosen_smoke_multiplier':chosen['multiplier'],'final_weight':final_weight,'trials':trials}
    write_json(path,result);print('CALIBRATED',result['weights'],flush=True)


if __name__=='__main__':main()
