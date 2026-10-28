"""Optimisers scored at the agent's own budgets. The agent overshoots its 100-design budget (it only checks between
rounds), so each agent trial is compared with random, TPE and GP-BO given exactly as many designs as that trial used.
Replays bench_live.run with the same seeds from the simulation cache (no new simulations if the replay is exact),
and checks it reproduces results/live_<c>.csv at 50/100/300 first. Filters are skipped: no optimiser run meets their
spec at any budget up to 300.
Run in WSL: PATH=~/as-env/bin:$PATH python budget_match.py CIRCUIT     output: results/budget_match_<c>.csv"""
import os
import sys

import optuna
import pandas as pd

import bench_live
from bench_live import HERE, Circuit, run

name = sys.argv[1]
optuna.logging.set_verbosity(optuna.logging.WARNING)
agent = pd.read_csv(os.path.join(HERE, 'results', f'agent_qwen3-8b-fixprompt_{name}.csv')).dropna(subset=['designs'])
live = pd.read_csv(os.path.join(HERE, 'results', f'live_{name}.csv')).set_index(['method', 'seed', 'budget'])
c, rows = Circuit(name), []
for method in ('random', 'tpe', 'gp'):
    for seed in range(5):
        n0 = len(c.cache)
        seq = run(c, method, seed)
        for b in bench_live.CHECKPOINTS:  # replay must match the logged run
            assert max(seq[:b], key=lambda m: m['nfom'])['ok_true'] == live.loc[(method, seed, b), 'ok_true'], (method, seed, b)
        for _, t in agent.iterrows():
            best = max(seq[:int(t.designs)], key=lambda m: m['nfom'])
            rows.append(dict(circuit=name, method=method, seed=seed, agent_trial=int(t.trial), budget=int(t.designs),
                             ok_true=best['ok_true'], agent_ok_true=t.ok_true))
        print(name, method, seed, 'new simulations', len(c.cache) - n0, flush=True)
pd.DataFrame(rows).to_csv(os.path.join(HERE, 'results', f'budget_match_{name}.csv'), index=False)
