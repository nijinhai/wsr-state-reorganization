"""Check headline claims against committed per-seed CSV/JSON, without Torch."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = ROOT/'results/wsr-v0.3-precompute'
    result = json.loads((base/'results.json').read_text())
    expected = {'A': (396, .3760850727558136, .3783637086550395),
                'B': (67404, .61572265625, .4342447916666667),
                'B-budget': (460748, .6351996660232544, .4805772602558136),
                'C2': (67404, .6270073850949606, .4450954894224803),
                'C4': (67404, .6359049677848816, .4588758746782939),
                'C8': (67404, .6443142294883728, .4592013955116272),
                'D8': (70604, .6345486044883728, .4494357605775197)}
    rows = list(csv.DictReader((base/'main-results.csv').open()))
    assert len(result['runs']) == 21
    for name, (parameters, seen, held) in expected.items():
        for group, value in [('seen_query', seen), ('held_query', held)]:
            measurements = []
            for seed in range(3):
                run_name = f'{name}-seed{seed}'
                run = result['runs'][run_name]
                assert run['parameters'] == parameters and run['updates'] == 2048
                score = run['test'][group]['all']['accuracy']
                matches = [r for r in rows if r['run'] == run_name and r['queries'] == group and r['hops'] == 'all']
                assert len(matches) == 1 and abs(float(matches[0]['accuracy'])-score) < 1e-10
                measurements.append(score)
            assert abs(statistics.mean(measurements)-value) < 1e-10
            assert abs(statistics.mean(measurements)-result['aggregate'][name][group]['mean']) < 1e-10
            assert abs(statistics.stdev(measurements)-result['aggregate'][name][group]['sd']) < 1e-10
        print(f'{name:8s} {parameters:7,d} params  seen={100*seen:.2f}% held={100*held:.2f}%')
    for seed in range(3):
        c = result['runs'][f'C8-seed{seed}']
        b = result['runs'][f'B-budget-seed{seed}']
        assert c['cost']['1']['prepare'] == b['cost']['1']['prepare'] == 458880
        assert c['test']['held_query']['all']['accuracy'] < b['test']['held_query']['all']['accuracy']
    manifest = json.loads((ROOT/'results/export-provenance.json').read_text())
    byte_identical = 0
    for row in manifest:
        path = ROOT/row['published_path']
        if path.suffix == '.md':
            continue  # Links are rewritten for the public directory structure.
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['original_sha256'], path
        byte_identical += 1
    print(f'PASS: 21 runs; headline/CSV/SD/MAC checks; {byte_identical} evidence files match original hashes.')


if __name__ == '__main__':
    main()
