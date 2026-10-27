"""Live optimiser comparison on one AMS-SizingBench circuit too big to enumerate: random search, Optuna TPE and
Optuna GP-BO, each for SEEDS seeds and a budget of 300 simulations, maximising AutoSizer's normalised FoM.
Simulations go through AutoSizer's own ngspice runner and are cached per design point (the 2,000-design random
sample is preloaded, new points are appended to results/live_cache_<c>.csv), so repeats cost nothing.
At budgets 50, 100 and 300 we score the best-FoM design found so far, as AutoSizer does, two ways:
  ok_true  : every spec in user_specs_metric is met (negative thresholds included)
  ok_autosizer : AutoSizer's check_user_specs_met(..., fom_only_check=False), which cannot parse negative thresholds
plus whether any fully feasible design was seen. Resumable per (method, seed).
Run in WSL with ngspice on PATH: PATH=~/as-env/bin:$PATH python bench_live.py CIRCUIT [SEEDS]      output: results/live_<c>.csv"""
import csv
import glob
import os
import sys

import numpy as np
import optuna
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
AS = os.path.join(HERE, 'third_party', 'AutoSizer')
sys.path.insert(0, AS)
from ground_truth import meets, norm_fom  # noqa: E402
from iterative_ota_optimization_test import ControlledOTAOptimizer  # noqa: E402
from utils.feedback_extraction import check_user_specs_met  # noqa: E402

PDK = os.path.expanduser('~/pdk/sky130_pdk/libs.tech/combined/sky130.lib.spice')
BUDGET = int(os.environ.get('BUDGET', 300))  # smaller only for smoke tests
CHECKPOINTS = [x for x in (50, 100, 300) if x <= BUDGET] or [BUDGET]


class Circuit:
    def __init__(self, name):
        os.chdir(AS)
        cfg = yaml.safe_load(open(os.path.join(AS, 'AMS-SizingBench', f'{name}.yaml')))
        cfg.update(pdk_lib_path=PDK, results_dir=os.path.expanduser(f'~/as-runs/live/{name}'))
        if 'pnp_model_path' in cfg:
            cfg['pnp_model_path'] = os.path.join(os.path.dirname(PDK), '..', '..', 'libs.ref', 'sky130_fd_pr', 'spice',
                                                 'sky130_fd_pr__pnp_05v5_W3p40L3p40.model.spice')
        self.opt, self.spec, self.name = ControlledOTAOptimizer(cfg), cfg['user_specs_metric'], name
        self.vars = self.opt.var_names
        self.choices = [cfg.get(f'{v}_values') or cfg['W_values'] for v in self.vars]
        self.cache_file = os.path.join(HERE, 'results', f'live_cache_{name}.csv')
        files = glob.glob(os.path.join(HERE, 'results', f'sample_{name}_[0-9].csv'))
        if os.path.exists(self.cache_file):
            files.append(self.cache_file)
        self.cache = {}
        for f in files:
            for r in pd.read_csv(f).to_dict('records'):
                self.cache[self.key([r[v] for v in self.vars])] = {k: v for k, v in r.items()
                                                                    if k not in self.vars and k not in ('meets_fom', 'meets_all')}

    @staticmethod
    def key(point):
        return tuple(round(float(v), 9) for v in point)

    def evaluate(self, point):
        """Metrics dict for a design point, with 'nfom' (normalised FoM), 'ok_true' and 'ok_autosizer'."""
        k = self.key(point)
        if k not in self.cache:
            res = {kk: v for kk, v in self.opt.simulate_ota_config(list(point)).results.items() if kk not in self.vars}
            self.cache[k] = res
            new = not os.path.exists(self.cache_file)
            with open(self.cache_file, 'a', newline='') as f:
                w = csv.DictWriter(f, [*self.vars, *res], extrasaction='ignore')
                if new:
                    w.writeheader()
                w.writerow({**dict(zip(self.vars, point)), **res})
        m = dict(self.cache[k])
        row = pd.DataFrame([m])
        try:
            nf = float(norm_fom(row, self.spec).iloc[0])
            m['ok_true'] = bool(meets(row, self.spec, pd.Series([nf])).iloc[0])
        except KeyError:  # simulation failed and returned no metrics
            nf, m['ok_true'] = np.nan, False
        m['nfom'] = nf if np.isfinite(nf) else -1e3
        m['ok_autosizer'] = bool(check_user_specs_met({**m, 'fom': m['nfom']}, self.spec, fom_only_check=False))
        return m


def run(c, method, seed):
    """Sequence of BUDGET evaluated designs proposed by the method."""
    rng, seen = np.random.default_rng(seed), []
    if method == 'random':
        return [c.evaluate([rng.choice(ch) for ch in c.choices]) for _ in range(BUDGET)]
    sampler = {'gp': lambda: optuna.samplers.GPSampler(seed=seed, n_startup_trials=10),
               'tpe': lambda: optuna.samplers.TPESampler(seed=seed, n_startup_trials=10)}[method]()
    study = optuna.create_study(direction='maximize', sampler=sampler)

    def objective(t):
        m = c.evaluate([ch[t.suggest_int(v, 0, len(ch) - 1)] for v, ch in zip(c.vars, c.choices)])
        seen.append(m)
        return m['nfom']

    study.optimize(objective, n_trials=BUDGET)
    return seen


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    name, seeds = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5
    c = Circuit(name)
    out = os.path.join(HERE, 'results', f'live_{name}.csv')
    done = set()
    if os.path.exists(out):
        done = {(r['method'], int(r['seed'])) for r in csv.DictReader(open(out))}
    for method in os.environ.get('METHODS', 'random tpe gp').split():  # split one circuit across processes
        for seed in range(seeds):
            if (method, seed) in done:
                continue
            seq = run(c, method, seed)
            rows = []
            for b in CHECKPOINTS:
                best = max(seq[:b], key=lambda m: m['nfom'])
                feas = [m['nfom'] for m in seq[:b] if m['ok_true']]
                rows.append(dict(circuit=name, method=method, seed=seed, budget=b, best_nfom=best['nfom'],
                                 ok_true=best['ok_true'], ok_autosizer=best['ok_autosizer'], any_feasible=bool(feas),
                                 best_feasible_nfom=max(feas) if feas else np.nan))
            new = not os.path.exists(out)
            with open(out, 'a', newline='') as f:
                w = csv.DictWriter(f, rows[0])
                if new:
                    w.writeheader()
                w.writerows(rows)
            print(f'{name} {method} seed {seed}: best {rows[-1]["best_nfom"]:.3g} ok_true {rows[-1]["ok_true"]} '
                  f'ok_autosizer {rows[-1]["ok_autosizer"]} cache {len(c.cache)}', flush=True)


if __name__ == '__main__':
    main()
