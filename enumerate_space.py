"""Exhaustive ground truth for a small AMS-SizingBench circuit: simulate every point of its discrete grid
(<var>_values for each variable if given, else W_values) with AutoSizer's own ngspice runner, and record each design's metrics, whether it
meets the FoM line alone (AutoSizer's default success check) and whether it meets every spec.
With a 4th argument N, simulate N distinct random grid points instead (seeded, same for every shard), for spaces
too big to enumerate. Run in WSL, one process per shard: ~/as-env/bin/python enumerate_space.py CIRCUIT SHARD NSHARDS [N]
Output: results/enum_<circuit>_<shard>.csv, or results/sample_<circuit>_<shard>.csv with N"""
import csv
import itertools
import operator
import os
import random
import re
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
AS = os.path.join(HERE, 'third_party', 'AutoSizer')
PDK = os.path.expanduser('~/pdk/sky130_pdk/libs.tech/combined/sky130.lib.spice')
sys.path.insert(0, AS)
os.chdir(AS)
from iterative_ota_optimization_test import ControlledOTAOptimizer  # noqa: E402

OPS = {'>': operator.gt, '<': operator.lt, '>=': operator.ge, '<=': operator.le}


def specs(text):
    """'fom > 0.5 AND dc_gain_db > 15' -> [('fom', '>', 0.5), ('dc_gain_db', '>', 15.0)]"""
    return [(m, op, float(v)) for m, op, v in re.findall(r'(\w+)\s*(>=|<=|>|<)\s*(-?[\d.eE+-]+)', text)]


def meets(res, rules):
    return all(res.get(m) is not None and OPS[op](res[m], v) for m, op, v in rules)


name, shard, nshards = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
cfg = yaml.safe_load(open(os.path.join(AS, 'AMS-SizingBench', f'{name}.yaml')))
cfg.update(pdk_lib_path=PDK, results_dir=os.path.expanduser(f'~/as-runs/enum/{name}_{shard}'))
if 'pnp_model_path' in cfg:  # bandgap: the bundled PDK has the pnp as a .model.spice file
    cfg['pnp_model_path'] = os.path.join(os.path.dirname(PDK), '..', '..', 'libs.ref', 'sky130_fd_pr', 'spice',
                                         'sky130_fd_pr__pnp_05v5_W3p40L3p40.model.spice')
opt = ControlledOTAOptimizer(cfg)
rules = specs(cfg['user_specs_metric'])
choices = [cfg.get(f'{v}_values') or cfg['W_values'] for v in opt.var_names]  # per-variable list if the YAML has one
if len(sys.argv) > 4:  # random sample of N distinct points, identical across shards
    rng, n, seen = random.Random(0), int(sys.argv[4]), set()
    while len(seen) < n:
        seen.add(tuple(rng.choice(c) for c in choices))
    grid, kind = sorted(seen)[shard::nshards], 'sample'
else:
    grid, kind = list(itertools.product(*choices))[shard::nshards], 'enum'

os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
out = os.path.join(HERE, 'results', f'{kind}_{name}_{shard}.csv')
t0 = time.time()
with open(out, 'w', newline='') as f:
    w = None
    for i, point in enumerate(grid):
        res = opt.simulate_ota_config(list(point)).results
        row = dict(zip(opt.var_names, point), **{k: v for k, v in res.items() if k not in opt.var_names},
                   meets_fom=int(meets(res, [r for r in rules if r[0] == 'fom'])), meets_all=int(meets(res, rules)))
        if w is None:
            w = csv.DictWriter(f, row)
            w.writeheader()
        w.writerow(row)
        if i % 50 == 0:
            f.flush()
            print(f'{name} shard {shard}: {i + 1}/{len(grid)} {time.time() - t0:.0f} s', flush=True)
print(f'{name} shard {shard}: done {len(grid)} in {time.time() - t0:.0f} s', flush=True)
