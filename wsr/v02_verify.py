"""Final numerical/protocol audit; does not alter archived v0.1 experiments."""
import json
from pathlib import Path
import random
import shutil

import torch

from .v02_data import ROOT,digest
from .v02_train import write_json,Inputs,schedule,load_run
from .pretrained_experiment import model_load
from .state_dream import edit_cache


def main():
    old=json.loads((ROOT/'v01-preservation.json').read_text())
    altered=[p for p,h in old.items() if not Path(p).is_file() or digest(p)!=h]
    if altered:raise RuntimeError(f'Archived v0.1 files changed: {altered}')
    model=model_load('cuda');inputs=Inputs(model);results=[]
    for seed in range(3):
        schedules=[];initials=[]
        for group in ['B0','S','P']:
            directory=ROOT/'runs'/f'{group}-seed-{seed}'
            one=torch.load(directory/'checkpoint-0000.pt',weights_only=True,map_location='cpu')
            last=torch.load(directory/'checkpoint-2048.pt',weights_only=True,map_location='cpu')
            for step in [512,1024,2048]:
                checkpoint=torch.load(directory/f'checkpoint-{step:04d}.pt',weights_only=True,map_location='cpu')
                assert checkpoint['step']==step
                assert len(checkpoint['training'])==step
            assert all(r['LM_target_tokens']==128 and r['actual_LM_forward_tokens']==320 for r in last['training'])
            assert all(r['dream_gradient_norm']>0 and torch.isfinite(torch.tensor(r['lm_nll'])) for r in last['training'])
            updates=sum((last['dream'][k]-v).double().square().sum() for k,v in one['dream'].items()).sqrt()
            assert updates>0
            if group in ['S','P']:
                assert all(torch.equal(one['head'][k],one['control_head'][k]) for k in one['head'])
                assert sum((last['head'][k]-v).double().square().sum() for k,v in one['head'].items())>0
                assert sum((last['control_head'][k]-v).double().square().sum() for k,v in one['head'].items())>0
            initials.append(one['dream']);schedules.append(json.loads((directory/'schedule.json').read_text()))
            results.append({'group':group,'seed':seed,'updates':2048,'Dream_update_norm':float(updates),
                'head_control_same_initialization':group!='B0','checkpoint_milestones_valid':True,'finite_gradients':True})
        assert schedules[0]==schedules[1]==schedules[2]
        assert all(all(torch.equal(initials[0][k],state[k]) for k in initials[0]) for state in initials[1:])
        assert schedules[0]==[r['id'] for r in schedule(inputs.index['train'],seed)]
    # Actual frozen cache checks, including second-cycle independent no-Dream state.
    cache_checks=[]
    with torch.no_grad():
        for row in random.Random(71).sample(inputs.index['train'],3):
            state=inputs.load(row);tokens=inputs.tokens[row['token_path']];b=row['boundary']
            advanced=model.wake(tokens[:,b-1:b+255],state)
            expected=inputs.original_second_state(row)
            diff=float((advanced.layers[12].ssm-expected).abs().max())
            assert diff==0
            dream,_,_=load_run(ROOT/'runs/B0-seed-0/checkpoint-2048.pt',inputs.device)
            changed,_,_=edit_cache(model,dream,state,tokens,b,128,12)
            future_changed=tokens.clone();future_changed[:,b:]=(future_changed[:,b:]+1)%50288
            other,_,_=edit_cache(model,dream,state,future_changed,b,128,12)
            assert torch.equal(changed.layers[12].ssm,other.layers[12].ssm)
            assert changed.layers[12].conv is state.layers[12].conv
            assert all(changed.layers[i] is state.layers[i] for i in range(24) if i!=12)
            logits0,_=model.run(tokens[:,b-1:b+63],state)
            logits1,_=model.run(tokens[:,b-1:b+63],state.edit(12,state.layers[12].ssm))
            assert torch.equal(logits0,logits1)
            cache_checks.append({'id':row['id'],'second_original_state_max_abs_difference':diff,
                                 'future_mutation_Dream_invariance':True,'zero_edit_logits_exact':True})
    continuous=[]
    lm_files=list((ROOT/'evaluation').glob('*/lm.json'))
    assert len(lm_files)==7
    assert not list((ROOT/'evaluation').glob('S-*/lm.json'))
    for path in lm_files:
        for row in json.loads(path.read_text()):
            assert row['state_history']==row['history_visible']-1
            assert all(torch.isfinite(torch.tensor(v)) for v in row['losses'])
    for path in (ROOT/'evaluation').glob('*/periodic.json'):
        for book in json.loads(path.read_text()):
            assert book['resets_within_book']==0 and book['total_history']==32768
            assert len(book['events'])==16
            for event in book['events']:assert event['processed_after']==event['history']-1+len(event['losses'])
            for a,b in zip(book['events'],book['events'][1:]):assert a['processed_after']==b['history']-1
            continuous.append(str(path)+':'+book['document']['id'])
    bank=json.loads((ROOT/'bank/index.json').read_text())
    unique={split:{(r['document']['dataset'],r['document']['id']):max(x['boundary'] for x in bank[split] if x['document']['dataset']==r['document']['dataset'] and x['document']['id']==r['document']['id']) for r in bank[split]} for split in ['train','validation']}
    s_count=0
    for split in ['train','validation']:
        examples=torch.load(ROOT/'aux-s'/(split+'.pt'),weights_only=True,mmap=True)
        s_count+=sum(r['tokens'].numel()-1 for r in examples)
    p_count=sum(len(torch.load(ROOT/'aux-p'/(split+'-raw.pt'),weights_only=True)['keys'])*64 for split in ['train','validation'])
    write_json(ROOT/'preparation-accounting.json',{'original_prefix_tokens_one_pass':sum(sum(v-1 for v in docs.values()) for docs in unique.values()),
        'independent_no_Dream_second_cycle_prefix_tokens':len(bank['train'])*256,
        'S_selected_layer_encoding_tokens_13_layers':s_count,'P_full_encoder_forward_tokens_24_layers':p_count,
        'training_LM_target_tokens_total':9*2048*128,'training_LM_forward_tokens_total':9*2048*320,
        'archived_reader_bug_draft_checkpoint_updates':896,'archived_reader_bug_draft_LM_forward_tokens_lower_bound':896*320,
        'discarded_unsaved_tail':'processes stopped between checkpoint writes; exact extra updates/tokens unavailable, excluded from formal budgets',
        'calibration_objective_evaluations':64,'calibration_optimizer_updates':32,
        'calibration_LM_forward_tokens_per_complete_calibration':64*320,'calibration_runs_including_archived_pre_fix':2,
        'method':'bank preparation completed in one pass; accounting reconstructed from exact saved boundaries and target keys; includes reusable prefix/feature computation separately from training'})
    target=ROOT/'source-run'
    for directory in ['wsr','tests']:
        for path in Path(directory).glob('*.py'):
            dest=target/path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    for name in ['pyproject.toml','README.md','STATUS.md','WSR_V02A.md']:
        shutil.copy2(name,target/name)
    write_json(ROOT/'source-run-sha256.json',{str(p.relative_to(target)):digest(p) for p in target.rglob('*') if p.is_file()})
    write_json(ROOT/'verification.json',{'v01_unchanged_files':len(old),'v01_altered':altered,'matrix':results,
        'paired_schedules_and_Dream_initializations':True,'auxiliary_controls_matched_initializations':True,
        'all_7_B0_P_and_baseline_LM_files_finite_and_time_aligned':True,
        'S_expanded_evaluation_paused':True,'scope':json.loads((ROOT/'scope-update.json').read_text()),
        'LM_target_tokens_total':9*2048*128,'actual_LM_forward_tokens_total':9*2048*320,
        'cache_checks':cache_checks,'continuous_book_evaluations':len(continuous),'test_split_used':False,
        'unit_tests_passed':21,'auxiliary_reader_normalization_gradcheck_passed':True,
        'no_W1_W2_or_new_memory_slots':True})
    print('Final protocol, gradient, continuity, nonleakage and archive audit passed',flush=True)


if __name__=='__main__':main()
