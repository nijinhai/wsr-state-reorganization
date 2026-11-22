"""Bounded experiment matrix; seed expansion gates numerical learning, not positive NLL."""
import json
from pathlib import Path
import subprocess
import sys
import time

from .v02_data import ROOT
from .v02_train import write_json


def run(*args):subprocess.run([sys.executable,'-m',*args],check=True)


def parallel(commands,label):
    """Two numerical workers on one GPU; file locks prevent duplicate run writes."""
    workers=[];streams=[]
    for i,args in enumerate(commands):
        dest=ROOT/f'{label}-worker-{i}.log';stream=dest.open('a');streams.append(stream)
        workers.append(subprocess.Popen([sys.executable,'-m',*args],stdout=stream,stderr=subprocess.STDOUT))
    last={}
    while any(p.poll() is None for p in workers):
        for i,p in enumerate(workers):
            path=ROOT/f'{label}-worker-{i}.log'
            lines=path.read_text().splitlines()
            line=lines[-1] if lines else 'initializing'
            if last.get(i)!=line:print(label,i,line,flush=True);last[i]=line
        time.sleep(20)
    for stream in streams:stream.close()
    if any(p.returncode for p in workers):raise RuntimeError(f'{label} numerical worker failed; completed runs/checkpoints retained; inspect worker logs')


def gate(groups):
    results=[]
    for group in groups:
        summary=json.loads((ROOT/'runs'/f'{group}-seed-0'/'summary.json').read_text())
        valid=summary['complete'] and summary['updates']==2048 and summary['base_frozen'] and summary['nonzero_finite_Dream_gradients']
        valid=valid and summary['dream_parameter_update_norm_this_invocation'] is not None and summary['dream_parameter_update_norm_this_invocation']>0
        valid=valid and summary['final_validation']['mean_nll']<100
        results.append({'group':group,'passed':valid,'criterion':'finite validation, actual Dream parameter updates and gradients, backbone frozen; NLL improvement not required'})
    if not all(r['passed'] for r in results):raise RuntimeError('Seed expansion numerical gate failed')
    write_json(ROOT/('seed-gate-'+'-'.join(groups)+'.json'),results)


def evaluate_matrix():
    plan=json.loads((ROOT/'execution-plan.json').read_text())
    plan['LM_evaluation_workers']=2
    plan['evaluation_shards']='worker0: baseline + B0/P seed0 (including 2-book periodic); worker1: B0/P seed1/2; S expanded evaluation paused'
    write_json(ROOT/'execution-plan.json',plan)
    parallel([('wsr.v02_evaluate','--groups','baseline','B0','P','--seeds','0'),
              ('wsr.v02_evaluate','--groups','B0','P','--seeds','1','2')],'evaluation')


def main():
    # Continuation scope: S results are retained, but no new S work is scheduled.
    gate(['B0','P'])
    parallel([('wsr.v02_train','--groups','B0','P','--seeds',str(seed)) for seed in [1,2]],'paired-continuation')
    evaluate_matrix()
    if not (ROOT/'feature-diagnostics.json').exists():run('wsr.v02_evaluate','--features-only')
    if not (ROOT/'matched-P-target-diagnostics.json').exists():run('wsr.v02_target_diagnostics')
    print('WSR-v0.2a B0/P CONTINUATION COMPLETE; S PAUSED',flush=True)


if __name__=='__main__':main()
