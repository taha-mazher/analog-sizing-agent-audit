"""Optimiser comparison on the enumerated AMS-SizingBench circuits, replayed from the ground-truth tables
(results/enum_<circuit>_*.csv), so every method sees exactly AutoSizer's simulator outputs and no new simulation runs.
Objective: AutoSizer's auto-generated normalised FoM (ground_truth.norm_fom), maximised, as in its 'fom_only' ranking.
Per run and budget we record the best FoM found, the regret to the true best, and two success definitions:
  fom : normalised FoM passes the spec's FoM line (AutoSizer's default fom_only_check)
  all : every spec passes
Methods: random with replacement, random without replacement, Optuna GP-BO and TPE. Every trial counts as one
simulation, repeats included. Usage: python bench.py [seeds] [budget]   output: results/bench_runs.csv"""
import glob
import os
import sys

import numpy as np
import optuna
import pandas as pd
import yaml

from ground_truth import BENCH, HERE, RULE, meets, norm_fom

SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
BUDGET = int(sys.argv[2]) if len(sys.argv) > 2 else 100
CHECKPOINTS = [b for b in (10, 25, 50, 100, 300) if b <= BUDGET]


def table(c):
    """Ground truth for one circuit: design-variable columns, normalised FoM and both success flags."""
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(HERE, 'results', f'enum_{c}_[0-9].csv')))])
    cfg = yaml.safe_load(open(os.path.join(BENCH, f'{c}.yaml')))
    spec, names = cfg['user_specs_metric'], list(cfg['variable'])
    d['nfom'] = norm_fom(d, spec).fillna(-np.inf)
    d['ok_all'] = meets(d, spec, d['nfom'])
    d['ok_fom'] = meets(d, ' AND '.join(f'{m} {op} {t}' for m, op, t in RULE.findall(spec) if m == 'fom') or 'fom > -1e300',
                        d['nfom'])
    choices = [sorted(d[n].unique()) for n in names]
    size = np.prod([len(cfg.get(f'{v}_values') or cfg['W_values']) for v in names])
    return (d.set_index(names), names, choices) if len(d) == size else None  # None: enumeration not finished


def replay(d, names, choices, method, seed):
    """Sequence of design rows the method asks for, BUDGET long."""
    rng = np.random.default_rng(seed)
    if method == 'random':
        return [d.loc[tuple(rng.choice(c) for c in choices)] for _ in range(BUDGET)]
    if method == 'random_noreplace':
        idx = rng.permutation(len(d))
        return [d.iloc[i] for i in np.resize(idx, BUDGET)]
    sampler = {'gp': lambda: optuna.samplers.GPSampler(seed=seed, n_startup_trials=10),
               'tpe': lambda: optuna.samplers.TPESampler(seed=seed, n_startup_trials=10)}[method]()
    study, rows = optuna.create_study(direction='maximize', sampler=sampler), []

    def objective(t):
        row = d.loc[tuple(c[t.suggest_int(n, 0, len(c) - 1)] for n, c in zip(names, choices))]
        rows.append(row)
        return float(row.nfom) if np.isfinite(row.nfom) else -1e3

    study.optimize(objective, n_trials=BUDGET)
    return rows


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    circuits = sorted({os.path.basename(f)[5:-6] for f in glob.glob(os.path.join(HERE, 'results', 'enum_*_[0-9].csv'))})
    out = []
    for c in circuits:
        t = table(c)
        if t is None:
            print(c, 'skipped: enumeration incomplete', flush=True)
            continue
        d, names, choices = t
        best = d.nfom[d.ok_all].max() if d.ok_all.any() else d.nfom.max()
        for method in ('random', 'random_noreplace', 'tpe', 'gp'):
            for seed in range(SEEDS):
                rows = pd.DataFrame(replay(d, names, choices, method, seed))
                for b in CHECKPOINTS:
                    r = rows.iloc[:b]
                    found = r.nfom[r.ok_all].max() if r.ok_all.any() else np.nan
                    out.append(dict(circuit=c, space=len(d), method=method, seed=seed, budget=b,
                                    success_fom=bool(r.ok_fom.any()), success_all=bool(r.ok_all.any()),
                                    best_feasible=found, regret_pct=100 * (best - found) / abs(best) if r.ok_all.any() else np.nan))
            print(c, method, 'done', flush=True)
    pd.DataFrame(out).to_csv(os.path.join(HERE, 'results', 'bench_runs.csv'), index=False)


if __name__ == '__main__':
    main()
