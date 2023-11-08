#!/bin/bash

# Import helper functions
source ../util/helper-functions.sh

# SIZES=( 2500 5000 7500 10000 12500 15000 17500 20000 )
SIZES=( 10000 )
# CPU_LEVELS=( 1 2 4 6 8 10 12 14 16 18 20 22 24 )
CPU_LEVELS=( 4 )

# Turn off all CPUs
disable_cpus "96"


for cpu_level in "${CPU_LEVELS[@]}"; do
  enable_cpus "$cpu_level"
  for size in "${SIZES[@]}"; do

    # Run the command and capture its output in a variable
    date=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    ../util/util-collector.sh $size matmult $cpu_level "$date" &
    power_meter_output=$(likwid-powermeter python3 matmult.py $size)

    # Extract runtime value (10.1151s)
    runtime=$(echo "$power_meter_output" | grep -oE 'Runtime: [0-9.]+ s' | awk '{print $2}')

    # Extract energy and power consumed for Domain PKG on each socket
    energy_socket_0=$(echo "$power_meter_output" | grep -A 4 'Measure for socket 0 on CPU 0' | grep 'Energy consumed:' | awk '{print $3}')
    power_socket_0=$(echo "$power_meter_output" | grep -A 4 'Measure for socket 0 on CPU 0' | grep 'Power consumed:' | awk '{print $3}')

    energy_socket_1=$(echo "$power_meter_output" | grep -A 4 'Measure for socket 1 on CPU 1' | grep 'Energy consumed:' | awk '{print $3}')
    power_socket_1=$(echo "$power_meter_output" | grep -A 4 'Measure for socket 1 on CPU 1' | grep 'Power consumed:' | awk '{print $3}')

    echo "$date,$size,$cpu_level,$runtime,$energy_socket_0,$power_socket_0,$energy_socket_1,$power_socket_1" >> data-energy-matmult.csv

    sleep 20

  done
done

# Turn on all cores before exiting
# enable_cpus "96"