"""Ground truth from the exhaustive enumerations (results/enum_<circuit>_*.csv).
For each circuit: how many designs meet the spec under the two FoMs AutoSizer has
  raw  : the testbench FoM from the circuit's YAML (metric_post.fom.expr), what enumerate_space.py recorded
  norm : the FoM AutoSizer's LLM pipeline auto-generates from user_specs_metric (_auto_generate_fom_expression)
and how likely random search is to find a top-1% design (full enumerations only; random samples, sample_*.csv, give
feasible fractions as estimates). Usage: python ground_truth.py [circuit ...]"""
import glob
import math
import os
import re
import sys

import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, 'third_party', 'AutoSizer', 'AMS-SizingBench')
RULE = re.compile(r'(\w+)\s*(>=|<=|>|<)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')


def norm_fom(d, spec):
    """Same terms as AutoSizer's _auto_generate_fom_expression: x/t for '> t', 1/(x/t) for '< t', abs() for negative t."""
    f = pd.Series(1.0, index=d.index)
    for m, op, t in RULE.findall(spec):
        t = float(t)
        if m == 'fom' or abs(t) < 1e-10:
            continue
        x = d[m].abs() / abs(t) if t < 0 else d[m] / t
        up = (op[0] == '>') == (t >= 0)          # numerator term
        f = f * x if up else f / x
    return f


def meets(d, spec, fom):
    ok = pd.Series(True, index=d.index)
    for m, op, t in RULE.findall(spec):
        x = fom if m == 'fom' else d[m]
        ok &= {'>': x > float(t), '<': x < float(t), '>=': x >= float(t), '<=': x <= float(t)}[op]
    return ok


def p_top1(n, k, b):
    """P(at least one of k top designs among b distinct random draws from n)."""
    return 1.0 if b >= n else 1 - math.comb(n - k, b) / math.comb(n, b)


def files(c):
    """Full enumeration if there is one, else the random sample."""
    return sorted(glob.glob(os.path.join(HERE, 'results', f'enum_{c}_[0-9].csv'))) or \
        sorted(glob.glob(os.path.join(HERE, 'results', f'sample_{c}_[0-9].csv')))


def main():
    names = sys.argv[1:] or sorted({os.path.basename(f).split('_', 1)[1][:-6] for f in
                                    glob.glob(os.path.join(HERE, 'results', '*_*_[0-9].csv'))})
    print(f"{'circuit':22s} {'designs':>7} {'raw ok':>7} {'norm ok':>8} {'norm FoM min..max':>20} {'P(top1%) 100/300 sims':>22}")
    for c in names:
        fs = files(c)
        d = pd.concat([pd.read_csv(f) for f in fs])
        sampled = 'sample_' in fs[0]
        spec = yaml.safe_load(open(os.path.join(BENCH, f'{c}.yaml')))['user_specs_metric']
        nf = norm_fom(d, spec)
        raw = d['fom'] if 'fom' in d else pd.Series(float('nan'), index=d.index)  # filter circuits report no raw FoM
        raw_ok, norm_ok = meets(d, spec, raw), meets(d, spec, nf)
        k = max(1, round(0.01 * len(d)))
        print(f"{c:22s} {len(d):7d} {100 * raw_ok.mean():6.1f}% {100 * norm_ok.mean():7.1f}% "
              f"{nf.min():9.3g}..{nf.max():<9.3g} " + ('   (random sample)' if sampled else
                                                       f"{p_top1(len(d), k, 100):9.2f} / {p_top1(len(d), k, 300):.2f}"))


if __name__ == '__main__':
    main()
