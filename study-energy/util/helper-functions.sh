#!/bin/bash

# Turn off all CPUs up to input
function disable_cpus {
  start_cpu=$1
  for ((cpu=1; cpu<$start_cpu; cpu++)); do
    echo 0 | sudo tee /sys/devices/system/cpu/cpu$cpu/online >/dev/null
  done
}

# Turn on all CPUs up to input number
function enable_cpus {
  start_cpu=$1
  for ((cpu=1; cpu<$start_cpu; cpu++)); do
    echo 1 | sudo tee /sys/devices/system/cpu/cpu$cpu/online >/dev/null
  done
}

# Turn on specific CPU number
function enable_cpu_no {
  cpu_no=$1
  echo 1 | sudo tee /sys/devices/system/cpu/cpu$cpu_no/online >/dev/null
}

# Turn off specificy CPU number
function disable_cpu_no {
  cpu_no=$1
  echo 0 | sudo tee /sys/devices/system/cpu/cpu$cpu_no/online >/dev/null
}