#!/bin/bash

SYSTEM=$1

# 1. Get PID of lachesis controller and kill the process
# lachesis_controller_pid=$(pgrep -f "python3 lachesis-controller.py")
# if [ -n "$lachesis_controller_pid" ]; then
#     kill $lachesis_controller_pid
#     echo "Killed lachesis controller"
# else
#     echo "No lachesis controller process"
# fi

# 2. Clear prediction table in lachesis database -- this will remove learned online models also
cd ~/lachesis/src/datastore
python3 datastore-lachesis.py
echo "Cleared prediction tables and removed learned online models"

# 3. Kill daemon processes, remove containers on invokers, and create CPU utilization plot
cd ~/lachesis/scripts/lachesis
./setup-workers.sh 3 $SYSTEM

