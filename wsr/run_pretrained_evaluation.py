"""Resume planned evaluation in bounded batches, preserving completed artifacts."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from .pretrained_report import NAMES,ROOT


def pending(phase):
    phases=['short','long-single','periodic','rounds'] if phase=='all' else [phase]
    jobs=[]
    for kind in phases:
        for name in NAMES:
            if kind=='rounds' and name in ['baseline','corrupt']:continue
            seeds=[0,1,2] if kind=='short' and name.isdigit() else [0]
            for seed in seeds:
                path=ROOT/'evaluation'/f'{name}-seed-{seed}'/(kind+'.json')
                if not path.exists():jobs.append((kind,name,seed))
    return jobs


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=['all','short','long-single','periodic','rounds'],default='all')
    p.add_argument('--max-jobs',type=int,default=4)
    args=p.parse_args()
    for kind,name,seed in pending(args.phase)[:args.max_jobs]:
        print('Running evaluation',kind,name,seed,flush=True)
        subprocess.run([sys.executable,'-u','-m','wsr.pretrained_evaluate',kind,'--name',name,'--seed',str(seed)],check=True)
    jobs=pending('all')
    (ROOT/'pending-evaluation.json').write_text(json.dumps(jobs,indent=2))
    print('Remaining planned evaluation jobs:',len(jobs),flush=True)

if __name__=='__main__':main()
