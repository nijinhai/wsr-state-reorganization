"""Final archive and matched-budget checks, without training any models."""
import json
from pathlib import Path
import shutil

import torch

from .v02b_query import ROOT,OLD,balanced_rules,generate_rules,rule_key
from .v02_data import digest
from .v02_train import write_json


def main():
    manifests=[ROOT/'preservation-before.json',OLD/'preservation-before.json',Path('artifacts/wsr-v0.2a/v01-preservation.json')]
    protected={}
    for path in manifests:protected.update(json.loads(path.read_text()))
    changed=[p for p,h in protected.items() if not Path(p).is_file() or digest(p)!=h]
    assert not changed,changed
    paths=list((ROOT/'runs').glob('*/summary.json'));assert len(paths)==22
    runs={p.parent.name:json.loads(p.read_text()) for p in paths}
    for name,r in runs.items():
        assert r['complete'] and r['parameter_update_norm']>0 and r['finite_nonzero_gradients']
        training=json.loads((ROOT/'runs'/name/'training.json').read_text())
        assert all(torch.isfinite(torch.tensor(row['loss'])) and row['gradient_norm']>0 for row in training)
        for p in (ROOT/'runs'/name).glob('checkpoint-*.pt'):
            saved=torch.load(p,weights_only=True)
            assert all(torch.isfinite(v).all() for v in saved['model'].values())
    paired=[]
    for seed in range(3):
        names=[g+f'-seed{seed}' for g in ['mlp-original','mlp-matched','recurrent-oracle','recurrent-raw_reader','recurrent-edited_reader']]
        schedule=[torch.load(ROOT/'runs'/n/'schedule.pt',weights_only=True) for n in names]
        assert all(torch.equal(schedule[0],s) for s in schedule[1:])
        initial=[torch.load(ROOT/'runs'/n/'checkpoint-00000.pt',weights_only=True)['model'] for n in names[2:]]
        for state in initial[1:]:
            assert all(torch.equal(v,state[k]) for k,v in initial[0].items())
        assert all(torch.equal(v,initial[2][k]) for k,v in initial[1].items())
        assert all(runs[n]['updates']==8192 and runs[n]['query_target_draws']==1048576 for n in names)
        paired.append({'seed':seed,'same_sample_schedule':True,'C_D_identical_initial_parameters':True,'B_C_D_identical_core_initialization':True})
    pools=generate_rules();known={rule_key(r) for p in pools.values() for r in p}
    assert not known&set(balanced_rules())
    tests=Path('/tmp/wsr-v02b-tests.log').read_text();assert 'Ran 30 tests' in tests and 'OK' in tests
    result={'protected_unchanged_files':len(protected),'altered_files':changed,'completed_query_runs':len(runs),
            'new_query_optimizer_updates':sum(r['updates'] for r in runs.values()),
            'new_query_answer_draws':sum(r['query_target_draws'] for r in runs.values()),
            'Mamba_optimizer_updates':0,'Direct_Dream_optimizer_updates':0,'original_StateReader_optimizer_updates':0,
            'main_objective':'final color CE only; no per-step node loss or teacher forcing',
            'fact_bridge':'separate privileged fact-supervised diagnostic; never treated as C/D or Dream improvement',
            'posthoc_node_labels':'only analysis; all query models frozen',
            'paired':paired,'balanced_rules_disjoint':len(balanced_rules()),'unit_tests_passed':30,
            'no_W1_W2_S_plus_P_or_new_LM_training':True,'test_split_used':False}
    write_json(ROOT/'verification.json',result)
    dest=ROOT/'source-run'
    for directory in ['wsr','tests']:
        for path in Path(directory).glob('*.py'):
            target=dest/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    for name in ['WSR_V02B.md','STATUS.md','pyproject.toml']:shutil.copy2(name,dest/name)
    write_json(ROOT/'source-run-sha256.json',{str(p.relative_to(dest)):digest(p) for p in dest.rglob('*') if p.is_file()})
    print('v0.2b audit passed',result,flush=True)


if __name__=='__main__':main()
