"""Every table in the README, from the result CSVs (no simulation). Usage: python summarize.py > results/summary.txt"""
import glob
import os
import sys

import pandas as pd
import yaml

from ground_truth import BENCH, files, meets, norm_fom

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'third_party', 'AutoSizer'))
from utils.feedback_extraction import check_user_specs_met  # noqa: E402

R = lambda pat: pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(HERE, 'results', pat))])
pct = lambda x: (100 * x).round(0).astype(int)

print('== 1. Replay on the enumerated circuits (results/bench_runs.csv), budget 100, 20 seeds: all-spec success %')
b = R('bench_runs.csv')
b = b[b.budget == 100]
print(pct(b.pivot_table(index='circuit', columns='method', values='success_all')).to_string())
print('\nmedian regret % of the best feasible design found:')
print(b.pivot_table(index='circuit', columns='method', values='regret_pct', aggfunc='median').round(1).to_string())

print('\n== 2. AutoSizer\'s own success check vs the real spec on the 2,000 random designs of each filter circuit')
for c in ('folded_cascode_ota_lp', 'folded_cascode_ota_hp', 'folded_cascode_ota_bandp'):
    d = pd.concat([pd.read_csv(f) for f in files(c)]).reset_index(drop=True)
    spec = yaml.safe_load(open(os.path.join(BENCH, f'{c}.yaml')))['user_specs_metric']
    nf = norm_fom(d, spec)
    true = meets(d, spec, nf)
    theirs = [check_user_specs_met({**r, 'fom': f}, spec, fom_only_check=False) for r, f in zip(d.to_dict('records'), nf)]
    print(f'{c:26s} AutoSizer check passes {100 * pd.Series(theirs).mean():5.1f}%   true all-spec {100 * true.mean():4.1f}%   '
          f'spec: {spec}')

print('\n== 3. Live comparison, 12 circuits x 3 methods x 5 seeds (results/live_*.csv): best-FoM design meets ...')
lv = R('live_[!c]*.csv')
for name, col in (('every spec (true)', 'ok_true'), ("AutoSizer's check", 'ok_autosizer'), ('any feasible design seen', 'any_feasible')):
    print(f'{name}, % of runs:')
    print(pct(lv.pivot_table(index='budget', columns='method', values=col)).to_string())
t = lv[lv.budget == 300]
print('\nper circuit at 300 simulations, % of 5 seeds (true / AutoSizer check):')
print(pct(t.pivot_table(index='circuit', columns='method', values='ok_true')).astype(str)
      .add(' / ').add(pct(t.pivot_table(index='circuit', columns='method', values='ok_autosizer')).astype(str)).to_string())
print(f'\nruns where the best-FoM design passes AutoSizer\'s check but fails the real spec: '
      f'{int((t.ok_autosizer & ~t.ok_true).sum())} of {len(t)}')
print(f'runs that saw a fully feasible design but reported an infeasible best-FoM one: '
      f'{int((t.any_feasible & ~t.ok_true).sum())} of {len(t)}')

print('\n== 4. The AutoSizer agent itself (results/agent_*.csv), released budget of 100 designs')
ag = []
for f in glob.glob(os.path.join(HERE, 'results', 'agent_*.csv')):
    d = pd.read_csv(f)
    d['run'] = os.path.basename(f)[6:].rsplit('_', 1)[0].split('_')[0]
    ag.append(d)
ag = pd.concat(ag)
ag['completed'] = ag.designs.notna()
ag['attempts_crashed'] = ag.crashed.fillna('').map(lambda s: len([x for x in s.split(' | ') if x]))
print(ag.groupby('run').agg(trials=('trial', 'size'), completed=('completed', 'sum'),
                            crashed_attempts=('attempts_crashed', 'sum'), claims_success=('ok_autosizer', 'sum'),
                            truly_meets_spec=('ok_true', 'sum'), median_designs=('designs', 'median'),
                            median_tokens_in=('tokens_in', 'median'), median_seconds=('seconds', 'median')).to_string())
print('\nper trial:')
print(ag[['run', 'circuit', 'trial', 'designs', 'api_calls', 'ok_autosizer', 'ok_true', 'seconds']].sort_values(
    ['run', 'circuit', 'trial']).to_string(index=False))

print('\n== 5. Optimisers at each agent trial\'s own budget (results/budget_match_*.csv, 5 seeds, replayed from cache)')
bm = R('budget_match_*.csv')
print(pct(bm.pivot_table(index=['circuit', 'agent_trial', 'budget', 'agent_ok_true'], columns='method', values='ok_true')).to_string())
print('all matched budgets, % of runs meeting every spec:', pct(bm.groupby('method').ok_true.mean()).to_dict())
