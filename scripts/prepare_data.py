"""Reproduce the compact browser dataset from the original UCI archive (stdlib)."""
import array
import collections
import hashlib
import json
import math
import pathlib
import random
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEED = 20261004
STEPS = 120
CHANNELS = ['PS1', 'PS2', 'PS3', 'FS1', 'FS2', 'EPS1']
CLASSES = [100, 90, 80, 73]

def main():
    archive = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT/'work/hydraulic.zip'
    z = zipfile.ZipFile(archive)
    profiles = [list(map(int, line.split())) for line in z.read('profile.txt').decode().splitlines()]
    runs = []
    for i, profile in enumerate(profiles):
        if not runs or runs[-1]['profile'] != profile[:4]:
            runs.append({'profile': profile[:4], 'cycles': []})
        runs[-1]['cycles'].append(i)
    rng = random.Random(SEED)
    selected = {s: [] for s in ['train', 'validation', 'test']}
    for label in CLASSES:
        group_ids = [i for i, g in enumerate(runs) if g['profile'][1] == label and len(g['cycles']) >= 5]
        rng.shuffle(group_ids)
        nval = max(6, int(len(group_ids)*.2))
        partition = {'validation': group_ids[:nval], 'test': group_ids[nval:2*nval], 'train': group_ids[2*nval:]}
        for split, ids in partition.items():
            candidates = [(cycle, g) for g in ids for cycle in runs[g]['cycles']]
            rng.shuffle(candidates)
            count = 100 if split == 'train' else 30
            assert len(candidates) >= count
            selected[split].extend(candidates[:count])
    for rows in selected.values():
        rng.shuffle(rows)
    wanted = {cycle for rows in selected.values() for cycle, _ in rows}
    compact = {i: [0.0]*(STEPS*len(CHANNELS)) for i in wanted}
    for channel, name in enumerate(CHANNELS):
        with z.open(name+'.txt') as stream:
            for cycle, line in enumerate(stream):
                if cycle not in wanted:
                    continue
                values = list(map(float, line.split()))
                assert len(values) % STEPS == 0
                block = len(values)//STEPS
                for t in range(STEPS):
                    compact[cycle][t*len(CHANNELS)+channel] = sum(values[t*block:(t+1)*block])/block
        print('Resampled', name, flush=True)
    train_ids = [i for i, _ in selected['train']]
    means = []
    scales = []
    for c in range(len(CHANNELS)):
        vals = [v for i in train_ids for v in compact[i][c::len(CHANNELS)]]
        mean = sum(vals)/len(vals)
        scale = math.sqrt(sum((v-mean)**2 for v in vals)/len(vals))
        means.append(mean)
        scales.append(max(scale, 1e-8))
    out = ROOT/'public/examples/hydraulic/data'
    out.mkdir(parents=True, exist_ok=True)
    metadata = {'version': 1, 'seed': SEED, 'steps': STEPS, 'durationSeconds': 60,
                'channels': CHANNELS, 'classes': CLASSES, 'task': 'Valve condition',
                'classNames': ['Optimal (100%)', 'Small lag (90%)', 'Severe lag (80%)', 'Near failure (73%)'],
                'normalization': {'fitOn': 'train', 'mean': means, 'scale': scales},
                'splitMethod': 'Seeded valve-stratified split of contiguous acquisition runs with identical first four profile columns; runs shorter than five cycles excluded; balanced cycle subsampling within split.',
                'source': {'title': 'Condition monitoring of hydraulic systems', 'authors': 'Helwig, Pignanelli, Schütze (2015)',
                           'doi': '10.24432/C5CW21', 'license': 'CC BY 4.0',
                           'uci': 'https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems',
                           'kaggle': 'https://www.kaggle.com/datasets/jjacostupa/condition-monitoring-of-hydraulic-systems',
                           'archiveSha256': hashlib.sha256(archive.read_bytes()).hexdigest()}, 'splits': {}}
    for split, rows in selected.items():
        data = array.array('f')
        for i, _ in rows:
            data.extend((v-means[j%len(CHANNELS)])/scales[j%len(CHANNELS)] for j,v in enumerate(compact[i]))
        if sys.byteorder != 'little':
            data.byteswap()
        payload = data.tobytes()
        (out/(split+'.bin')).write_bytes(payload)
        metadata['splits'][split] = {'count': len(rows), 'file': split+'.bin',
                                    'sha256': hashlib.sha256(payload).hexdigest(),
                                    'labels': [CLASSES.index(profiles[i][1]) for i,_ in rows],
                                    'cycleIds': [i+1 for i,_ in rows], 'runIds': [g for _,g in rows],
                                    'allProfiles': [profiles[i] for i,_ in rows]}
    groups = [set(v['runIds']) for v in metadata['splits'].values()]
    assert not (groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print({s: (v['count'],dict(collections.Counter(v['labels']))) for s,v in metadata['splits'].items()})

if __name__ == '__main__':
    main()
