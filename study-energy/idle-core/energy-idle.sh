#!/bin/bash

# Import helper functions
source ../util/helper-functions.sh

NO_CPUS=48

# Turn off all CPUs
disable_cpus "96"

for ((cpu=0; cpu<$NO_CPUS; cpu+=2)); do
    one_more=$((cpu+1))
    enable_cpu_no "$cpu"
    enable_cpu_no "$one_more"

    for ((run_no=0; run_no<10; run_no++)); do
        turbostat_output=$(sudo turbostat --quiet --show PkgWatt --interval 5 --num_iterations 1 | tr '\n' ' ')
        total_pwr=$(echo "$turbostat_output" | cut -d' ' -f2)
        pwr_s0=$(echo "$turbostat_output" | cut -d' ' -f3)
        pwr_s1=$(echo "$turbostat_output" | cut -d' ' -f4)
        echo "$run_no,$cpu,$total_pwr,$pwr_s0,$pwr_s1" >> ~/lachesis/data/energy-study/data-energy-idle.csv
    done
    echo "Completed CPU $cpu"
done