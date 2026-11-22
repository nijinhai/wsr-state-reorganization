"""Report per-family accuracy across worlds, separately for every run."""
import json
import sys
from collections import defaultdict

for run in json.load(open(sys.argv[1])):
    groups=defaultdict(list)
    for row in run['metrics']:
        groups[row['family']].append(row)
    print(f"config={run['config']:02d} seed={run['seed']}")
    for family,rows in groups.items():
        accuracy=sum(row['correct'] for row in rows)/len(rows)
        loss=sum(row['loss'] for row in rows)/len(rows)
        print(f'  {family}: accuracy={accuracy:.3f} loss={loss:.4f} episodes={len(rows)}')
