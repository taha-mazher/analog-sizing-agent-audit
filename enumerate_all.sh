#!/bin/bash
# Exhaustive enumeration of the small AMS-SizingBench circuits, two shards at a time on cores 10 and 11.
# Run in WSL from the project folder: bash enumerate_all.sh [circuit ...]   log: results/enum_all.log
cd "$(dirname "$0")"
export PATH=$HOME/as-env/bin:$PATH
mkdir -p results
for c in "${@:-inverter resistive_load_amp diode_load_amp nand_gate xor_gate current_mirror_ota telescopic_ota buffer}"; do
    for c1 in $c; do
        echo "start $c1 $(date +%T)" >> results/enum_all.log
        taskset -c 10 nice -n 10 python enumerate_space.py $c1 0 2 > results/enum_${c1}_0.log 2>&1 &
        taskset -c 11 nice -n 10 python enumerate_space.py $c1 1 2 > results/enum_${c1}_1.log 2>&1
        wait
        tail -qn 1 results/enum_${c1}_0.log results/enum_${c1}_1.log >> results/enum_all.log
    done
done
echo "ENUM_DONE $(date +%T)" >> results/enum_all.log
