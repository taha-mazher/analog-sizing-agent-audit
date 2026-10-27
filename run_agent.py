"""Run the AutoSizer LLM agent itself (its released run_llm_guided_optimization, main.py settings except the budget)
on one AMS-SizingBench circuit for several trials, then score its best design two ways, as bench_live.py does:
  ok_true      : every spec in user_specs_metric is met (negative thresholds included)
  ok_autosizer : AutoSizer's own specs_met, which drops negative thresholds
plus designs simulated, evals to best, tokens, API calls, wall time and the crashes of failed attempts
(up to ATTEMPTS per trial, default 3). Resumable per trial.
LLM: MODEL env (default qwen3:8b) served by any OpenAI-compatible server at LOCAL_LLM_URL; gemini-* uses GOOGLE_API_KEY.
Run in WSL: PATH=~/as-env/bin:$PATH LOCAL_LLM_URL=http://localhost:11434/v1 python run_agent.py CIRCUIT [TRIALS] [BUDGET]
output: results/agent_<model>_<circuit>.csv"""
import csv
import json
import os
import sys

import pandas as pd
import yaml

from bench_live import AS, PDK
from ground_truth import meets, norm_fom

sys.path.insert(0, AS)
from llm_guided_ota_optimization_test import run_llm_guided_optimization  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get('MODEL', 'qwen3:8b')
TAG = MODEL.replace(':', '-').replace('/', '-') + ('-fixprompt' if os.getenv('AUDIT_FIX_PROMPT') else '')
ATTEMPTS = int(os.environ.get('ATTEMPTS', 3))  # tries per trial; the crashes are logged


def main():
    name = sys.argv[1]
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    budget = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    tag = TAG
    out = os.path.join(HERE, 'results', f'agent_{tag}_{name}.csv')
    done = {int(r['trial']) for r in csv.DictReader(open(out))} if os.path.exists(out) else set()
    os.chdir(AS)
    for t in range(trials):
        if t in done:
            continue
        cfg = yaml.safe_load(open(os.path.join(AS, 'AMS-SizingBench', f'{name}.yaml')))
        rdir = os.path.expanduser(f'~/as-runs/agent/{tag}/{name}_trial_{t}')
        cfg.update(pdk_lib_path=PDK, results_dir=rdir)
        if 'pnp_model_path' in cfg:
            cfg['pnp_model_path'] = os.path.join(os.path.dirname(PDK), '..', '..', 'libs.ref', 'sky130_fd_pr', 'spice',
                                                 'sky130_fd_pr__pnp_05v5_W3p40L3p40.model.spice')
        # main.py settings: 3 variables, 3 regeneration cycles, plateau patience 2, pre-layout only
        errors = []
        for attempt in range(ATTEMPTS):  # the agent crashes when the LLM's JSON breaks its expected schema; retry
            try:
                run_llm_guided_optimization(config=cfg, max_total_designs=budget, num_variables_to_optimize=3,
                                            pre_layout_only=True, max_regeneration_cycles=3, plateau_patience=2,
                                            trial_index=t, model=MODEL)
                s = json.load(open(os.path.join(rdir, f'trial_{t}_summary.json')))
                break
            except Exception as e:
                if 'TooManyRequests' in repr(e) or 'RateLimit' in repr(e):  # API quota, not the agent: stop, rerun later
                    sys.exit(f'{name} trial {t}: API quota hit, stopping without recording: {str(e)[:300]}')
                print(f'{name} trial {t} attempt {attempt}: CRASHED {e!r}', flush=True)
                errors.append(repr(e)[:120])
        else:  # every attempt crashed: a failed trial
            s = dict(best_design_metrics={}, specs_met=False, token_usage={})
        s['crashed'] = ' | '.join(errors)
        spec, m = cfg['user_specs_metric'], pd.DataFrame([s['best_design_metrics']])
        try:
            ok_true = bool(meets(m, spec, norm_fom(m, spec)).iloc[0])
        except KeyError:  # best design is missing a spec metric
            ok_true = False
        tok = s['token_usage']
        row = dict(circuit=name, model=MODEL, trial=t, budget=budget, designs=s.get('total_designs_searched'),
                   evals_to_best=s.get('evals_to_best'), best_fom=s.get('best_fom'), ok_autosizer=s['specs_met'],
                   ok_true=ok_true, tokens_in=tok.get('total_input_tokens'), tokens_out=tok.get('total_output_tokens'),
                   api_calls=tok.get('total_api_calls'), seconds=round(s.get('total_time_seconds', 0)),
                   crashed=s.get('crashed', ''))
        new = not os.path.exists(out)
        with open(out, 'a', newline='') as f:
            w = csv.DictWriter(f, row)
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"{name} trial {t}: ok_true {ok_true} ok_autosizer {s['specs_met']} designs {row['designs']} "
              f"tokens {row['tokens_in']}/{row['tokens_out']} {row['seconds']} s", flush=True)


if __name__ == '__main__':
    main()
