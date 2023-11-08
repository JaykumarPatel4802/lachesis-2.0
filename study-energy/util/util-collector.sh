#!/bin/bash

# Check if the script receives the required number of arguments
if [ $# -ne 4 ]; then
  echo "Usage: $0 <input_size> <app> <no_cores> <date>"
  exit 1
fi

input_size="$1"
app="$2"
no_cores="$3"
date="$4"
process="python3 $app.py $input_size"

# Search for the PID of the process we want to get utilization for
while true; do
    # Find the PID corresponding to the command (second line of output)
    pid=$(ps -aux | grep "$process" | grep -v "grep" | grep -v "/usr/local/bin" | awk '{print $2}')
    
    # Check if the PID is not empty (command found)
    if [ -n "$pid" ]; then
        break
    fi
done

while true; do
    # Run the top command, redirect stderr to stdout, and extract the CPU utilization using awk
    cpu_util=$(top -b -n 2 -d 0.2 -p "$pid" 2>&1 | tail -1 | awk '{print $9}')

    # Check if cpu_util contains "signal 11"
    if [[ "$cpu_util" == "%CPU" ]]; then
        # echo "Process completed (Signal 11 encountered)"
        exit 0
    fi

    echo "$date,$input_size,$no_cores,$cpu_util" >> ../$app/data-util-$app.csv
done




