#!/bin/bash
# AutoSizer agent on the 6 circuits that matter (3 filters with the dropped negative specs, 3 where TPE/GP only partly
# succeed), 3 trials each at the released budget of 100. 3 circuits at a time. MODEL and keys come from .env / env.
cd "$(dirname "$0")"
set -a; . <(tr -d '\r' < .env); set +a
export PATH=$HOME/as-env/bin:$PATH MODEL=${MODEL:-gemini-3.5-flash}
C=(folded_cascode_ota_lp iac folded_cascode_ota_hp nmcnr folded_cascode_ota_bandp smc)
for k in 0 1 2; do
    ( for ((i = k; i < ${#C[@]}; i += 3)); do
          taskset -c $((4 + k)) python -W ignore run_agent.py ${C[$i]} 3 100 >> results/agent_${MODEL//[:\/]/-}_${C[$i]}.log 2>&1
      done ) &
done
wait
echo "AGENT_DONE $MODEL $(date +%T)" >> results/agent_all.log
