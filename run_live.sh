#!/bin/bash
# Live optimiser comparison on the 12 large circuits (VCO skipped: 40 s per simulation), 5 seeds x 3 methods x 300
# simulations each. 4 worker slots on cores 4-7, each taking every 4th circuit. A guard stops all workers if Windows
# free memory drops below 1.5 GB; everything is resumable per (circuit, method, seed), so just rerun this script.
cd "$(dirname "$0")"
export PATH=$HOME/as-env/bin:$PATH
PS=/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe
free_mb() { $PS -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1024)" | tr -d '\r'; }
C=(azc nmcnr folded_cascode_ota_lp iac fdgb ldo_regulator folded_cascode_ota_hp smc dfcfc 3_stage_ring_osc folded_cascode_ota_bandp folded_cascode_ota)
rm -f results/.stop
echo "start $(date +%T)" >> results/live_all.log
for k in 0 1 2 3; do
    ( for ((i = k; i < ${#C[@]}; i += 4)); do
          [ -e results/.stop ] && break  # guard tripped: don't start the next circuit
          taskset -c $((4 + k)) nice -n 5 python bench_live.py ${C[$i]} 5 >> results/live_${C[$i]}.log 2>&1
          [ -e results/.stop ] || echo "done ${C[$i]} $(date +%T)" >> results/live_all.log
      done ) &
done
while pgrep -f bench_live.py > /dev/null; do
    if [ "$(free_mb)" -lt 1536 ]; then touch results/.stop; pkill -f bench_live.py; echo "GUARD: stopped, low memory, $(date +%T)" >> results/live_all.log; break; fi
    sleep 20
done
wait
echo "LIVE_DONE $(date +%T)" >> results/live_all.log
