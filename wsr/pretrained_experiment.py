"""WSR-v0.1 on frozen official Mamba2-130M; no synthetic task training."""
import argparse
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .pretrained_model import PretrainedMamba2
from .pretrained_data import documents
from .state_dream import StateDream,damage,edit_cache,history_input,trace_metrics

ROOT=Path('artifacts/pretrained-v1')
CASES={f'{i+1:02d}':{'mode':mode,'training':training,'corruption':corruption}
       for i,(mode,training,corruption) in enumerate((m,t,c) for m in ['A','D'] for t in ['T1','T2'] for c in [False,True])}


def model_load(device):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    return PretrainedMamba2.load(device=device)


def continuation(model,tokens,cache,boundary,length):
    if boundary<1 or boundary+length>tokens.shape[1]: raise ValueError('Continuation must stay within one document')
    if cache.processed!=boundary-1: raise ValueError('Stale first-token prediction would be used')
    logits,state=model.run(tokens[:,boundary-1:boundary+length-1],cache)
    target=tokens[:,boundary:boundary+length]
    loss=F.cross_entropy(logits.flatten(0,1),target.flatten())
    return loss,state


def pick_training(step,seed,minimum):
    dataset=['wikitext103','pg19'][step%2]
    pool=[r for r in documents(dataset,'train') if len(r['tokens'])>=minimum]
    if not pool: raise ValueError('Insufficient real training documents')
    return pool[(step//2+seed)%len(pool)]


def train_case(model,case,seed,steps,denoise,output,history_window=128,layer=12,horizon=64,cycles=2,interval=256):
    config={**CASES[case],'case':case,'seed':seed,'prediction_steps':steps,'denoise_steps':denoise if CASES[case]['training']=='T2' else 0,
            'history_window':history_window,'layer':layer,'gradient_horizon':horizon,'cycles':cycles,'interval':interval,
            'rounds':8,'hidden':64,'lr':.0003,'dtype':'float32','precision':'full FP32','base_model':model.asset_manifest,
            'history_lengths':[512,1024,2048], 'bptt':'detach between cycles; no gradient across previous wake segments'}
    if CASES[case]['training']=='T2': config['prediction_schedule']='phase-local; same supervised documents/boundaries/masks as T1'
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    configfile=output/'config.json'
    if configfile.exists() and json.loads(configfile.read_text())!=config: raise ValueError('Refuse to overwrite a different run')
    configfile.write_text(json.dumps(config,indent=2))
    if (output/'summary.json').exists() and (output/'dream.pt').exists():
        summary=json.loads((output/'summary.json').read_text())
        if summary.get('completed'):
            dream,_=load_dream(output,next(model.parameters()).device)
            print('Preserved completed training',output.name,flush=True)
            return dream,summary
    torch.manual_seed(seed)
    dream=StateDream(mode=config['mode']).to(next(model.parameters()).device)
    optimizer=torch.optim.Adam(dream.parameters(),lr=config['lr'])
    versions=[p._version for p in model.parameters()]
    done=0;rows=[]
    if (output/'progress.pt').exists():
        saved=torch.load(output/'progress.pt',map_location=next(model.parameters()).device,weights_only=True)
        dream.load_state_dict(saved['dream']);optimizer.load_state_dict(saved['optimizer'])
        done=saved['completed_steps'];rows=saved['rows']
    total=config['denoise_steps']+steps
    start=time.monotonic()
    for step in range(done,total):
        is_denoise=step<config['denoise_steps']
        phase_step=step if is_denoise else step-config['denoise_steps']
        torch.manual_seed(seed*100000+phase_step+(1000000 if is_denoise else 0))
        boundary=[512,1024,2048][phase_step%3]
        record=pick_training(phase_step,seed,boundary+cycles*interval+horizon+1)
        tokens=record['tokens'][None].to(next(model.parameters()).device)
        state=model.wake(tokens[:,:boundary-1])
        optimizer.zero_grad(set_to_none=True)
        losses=[];metrics=[];raw=[]
        for cycle in range(1 if is_denoise else cycles):
            original=state.layers[layer].ssm
            use_corruption=True if is_denoise else config['corruption']
            edited,trace,initial=edit_cache(model,dream,state,tokens,boundary,history_window,layer,
                                            corruption=use_corruption,seed=10000+seed*10000+phase_step*cycles+cycle+(1000000 if is_denoise else 0))
            if is_denoise:
                loss=F.mse_loss(edited.layers[layer].ssm,original.detach())
                advanced=edited
            else:
                # Frozen parameters, but no no_grad around this continuation.
                loss,advanced=continuation(model,tokens,edited,boundary,horizon)
            losses.append(loss)
            metrics.append({'cycle':cycle,'history_visible':boundary,'state_history':state.processed,
                            'loss':float(loss.detach()),'trace':trace_metrics(initial,trace)})
            if step in [0,total-1]:
                raw.append({'boundary':boundary,'original':original.detach().cpu(),'input':initial.detach().cpu(),
                            'states':[h.detach().cpu() for h in trace]})
            state=advanced.detach()
            if not is_denoise and cycle+1<cycles:
                next_boundary=boundary+interval
                # The unscored part is ordinary wake, never a Dream input before it occurs.
                state=model.wake(tokens[:,state.processed:next_boundary-1],state)
                boundary=next_boundary
        objective=torch.stack(losses).mean()
        if not torch.isfinite(objective):raise RuntimeError('Nonfinite Dream loss')
        objective.backward()
        gradient=torch.nn.utils.clip_grad_norm_(dream.parameters(),1.,error_if_nonfinite=True)
        if float(gradient)==0:raise RuntimeError('Dream has no loss gradient')
        optimizer.step()
        assert all(p.grad is None and not p.requires_grad for p in model.parameters())
        assert [p._version for p in model.parameters()]==versions
        row={'step':step+1,'phase':'denoise' if is_denoise else 'prediction','document':record['metadata'],
             'objective':float(objective.detach()),'gradient_norm':float(gradient),'cycles':metrics}
        rows.append(row)
        if raw:torch.save(raw,output/f'edit-trace-step-{step+1:04d}.pt')
        if (step+1)%4==0 or step+1==total:
            torch.save({'dream':dream.state_dict(),'optimizer':optimizer.state_dict(),'config':config,
                        'completed_steps':step+1,'rows':rows},output/'progress.pt')
            (output/'training.json').write_text(json.dumps(rows,indent=2))
            print(output.name,step+1,'/',total,'loss',round(float(objective.detach()),5),'grad',round(float(gradient),6),flush=True)
    torch.save({'dream':dream.state_dict(),'config':config},output/'dream.pt')
    summary={'config':config,'completed':True,'actual_steps':len(rows),'all_base_weights_unchanged':True,
             'finite_nonzero_gradients':all(r['gradient_norm']>0 for r in rows),
             'seconds_this_invocation':time.monotonic()-start,
             'first_prediction_loss':next(r['objective'] for r in rows if r['phase']=='prediction'),
             'last_prediction_loss':rows[-1]['objective']}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    return dream,summary


def load_dream(path,device):
    saved=torch.load(Path(path)/'dream.pt',map_location=device,weights_only=True)
    dream=StateDream(mode=saved['config']['mode']).to(device)
    dream.load_state_dict(saved['dream']);dream.eval()
    return dream,saved['config']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['smoke','pilot','train'])
    p.add_argument('--case',choices=CASES,default='05')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--steps',type=int,default=32)
    p.add_argument('--denoise',type=int,default=8)
    p.add_argument('--history-window',type=int,default=128)
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    verification=json.loads((ROOT/'interface-verification.json').read_text())
    if not verification['passed']:raise ValueError('Interface verification must pass first')
    model=model_load(args.device)
    if args.phase=='smoke':
        results=[]
        for case in ['01','04','05','08']:
            _,summary=train_case(model,case,0,2,2,ROOT/'smoke'/case,horizon=32,cycles=1)
            results.append(summary)
        (ROOT/'smoke.json').write_text(json.dumps({'passed':all(r['finite_nonzero_gradients'] for r in results),'results':results},indent=2))
    elif args.phase=='pilot':
        if not json.loads((ROOT/'smoke.json').read_text())['passed']:raise ValueError('Smoke must pass')
        dream,summary=train_case(model,'05',0,12,0,ROOT/'pilot',horizon=64,cycles=2)
        # Gate measures actual state-to-prediction effect, never a short-term gain threshold.
        record=documents('wikitext103','validation')[0]
        tokens=record['tokens'][None].to(args.device);boundary=512
        state=model.wake(tokens[:,:boundary-1])
        with torch.no_grad():
            modified,trace,initial=edit_cache(model,dream,state,tokens,boundary,128,12)
            before,_=continuation(model,tokens,state,boundary,128)
            after,_=continuation(model,tokens,modified,boundary,128)
        relative=float((modified.layers[12].ssm-initial).norm()/initial.norm())
        result={'passed':summary['finite_nonzero_gradients'] and relative>1e-6,
                'validation_baseline_nll':float(before),'validation_dream_nll':float(after),
                'relative_edit':relative,'criterion':'finite training gradients and nonzero learned state edit; no short-term gain filter',
                'training':summary}
        (ROOT/'pilot.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
    else:
        if not json.loads((ROOT/'pilot.json').read_text())['passed']:raise ValueError('Pilot interface gate must pass')
        suffix='' if args.history_window else '-state-only'
        path=ROOT/'formal'/f'{args.case}-seed-{args.seed}{suffix}'
        train_case(model,args.case,args.seed,args.steps,args.denoise,path,history_window=args.history_window)

if __name__=='__main__':main()
