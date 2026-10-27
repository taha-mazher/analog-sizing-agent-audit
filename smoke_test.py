"""Smoke test: one AMS-SizingBench circuit simulated through AutoSizer's own ngspice runner, no LLM.
Points the YAML at the SKY130 models bundled with AnalogGym-Opt, copied into WSL (~/pdk). Run in WSL: ~/as-env/bin/python smoke_test.py [circuit]"""
import os
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
AS = os.path.join(HERE, 'third_party', 'AutoSizer')
PDK = os.path.expanduser('~/pdk/sky130_pdk/libs.tech/combined/sky130.lib.spice')  # copy of AnalogGym-Opt's bundled models; reading them over /mnt/c is ~20x slower
sys.path.insert(0, AS)
os.chdir(AS)  # the runner resolves circuit_sim helpers relative to the repo
from iterative_ota_optimization_test import ControlledOTAOptimizer  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else 'five_trans_ota'
cfg = yaml.safe_load(open(os.path.join(AS, 'AMS-SizingBench', f'{name}.yaml')))
cfg.update(pdk_lib_path=PDK, results_dir=os.path.expanduser(f'~/as-runs/smoke/{name}'))
opt = ControlledOTAOptimizer(cfg)
W = cfg['W_values']
for trial in ([W[0]] * len(opt.var_names), [W[len(W) // 2]] * len(opt.var_names), [W[-1]] * len(opt.var_names)):
    t0 = time.time()
    r = opt.simulate_ota_config(trial)
    print(f'{name} {trial} -> {r} ({time.time() - t0:.2f} s)')
