#!/bin/bash
# After the enumeration (results/enum_all.log ENUM_DONE), estimate how hard the circuits too big to enumerate are:
# N random designs each (2,000; 200 for the slow VCO). The Vivado BO run has priority: 2 shards on cores 10-11 while
# it runs, 4 shards (cores 8-11) once out/bo_run.log says RUN_DONE and results/VIVADO_BUSY does not exist. If Windows free memory drops below 1.5 GB, the
# shards are stopped and the circuit is retried with half the shards (down to 1) once it is back above 2.5 GB. Run from Git Bash; log results/sample_all.log.
# Skipped: bandgap_reference (missing pnp subcircuit file in the release) and switched_capacitor (ngspice error).
P=/c/Users/taham/analog-sizing
LOG=$P/results/sample_all.log
BOLOG="/c/Users/taham/OneDrive/Desktop/Human Activity Classification/hw_fix/out/bo_run.log"
free_mb() { powershell -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1024)" | tr -d '\r'; }
until grep -q ENUM_DONE $P/results/enum_all.log 2>/dev/null; do sleep 60; done
for c in azc ldo_regulator nmcnr 3_stage_ring_osc iac smc fdgb dfcfc folded_cascode_ota folded_cascode_ota_lp \
         folded_cascode_ota_hp folded_cascode_ota_bandp voltage_controlled_osc; do
    N=2000; [ $c = voltage_controlled_osc ] && N=200
    [ -f $P/results/sample_${c}.done ] && continue
    if grep -q RUN_DONE "$BOLOG" && [ ! -f $P/results/VIVADO_BUSY ]; then S=4; else S=2; fi  # 8 ran out of memory next to Chrome
    while :; do
        until [ "$(free_mb)" -gt 2500 ]; do sleep 30; done
        rm -f $P/results/sample_${c}_*.csv
        echo "start $c N=$N shards=$S $(date +%T)" >> $LOG
        for ((k = 0; k < S; k++)); do
            wsl -e bash -lc "cd /mnt/c/Users/taham/analog-sizing && PATH=\$HOME/as-env/bin:\$PATH taskset -c $((11 - k)) nice -n 10 python enumerate_space.py $c $k $S $N > results/sample_${c}_$k.log 2>&1" &
        done
        stopped=0
        while [ -n "$(jobs -r)" ]; do
            if [ "$(free_mb)" -lt 1536 ]; then
                wsl -e pkill -f "enumerate_space.py $c "
                stopped=1
                echo "GUARD: stopped $c, low memory, $(date +%T)" >> $LOG
            fi
            sleep 15
        done
        wait
        [ $stopped = 0 ] && break
        [ $S -gt 1 ] && S=$((S / 2))  # retry with fewer parallel simulations
    done
    tail -qn 1 $P/results/sample_${c}_*.log >> $LOG
    touch $P/results/sample_${c}.done
done
echo "SAMPLE_DONE $(date +%T)" >> $LOG
