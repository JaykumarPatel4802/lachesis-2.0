#!/bin/bash

SYSTEM=$1

# 0. Kill running daemons
echo "Killing running daemons"
./setup-workers.sh 3 $SYSTEM
pkill -f greenroute-controller.py

# 1. Clear prediction table in lachesis database -- this will remove learned online models also
cd ~/lachesis-2.0/src/datastore
python3 datastore-lachesis.py
echo "Cleared prediction tables and removed learned online models"

cd ~/lachesis-2.0/src
# python3 greenroute-controller.py & # no debug
tmux new-session -d -s lachesis 'python3 greenroute-controller.py' # debug
echo "Started greenroute controller"

# 2. Setup lachesis daemons
echo "Setting up workers"

cd ~/lachesis-2.0/scripts/lachesis
./setup-workers.sh 0 $SYSTEM

# 3. Redeploy OpenWhisk controller - remove any state
cd ~/lachesis-2.0/scripts/ow-scripts
./wsk-redeploy-component.sh controller

# 4. Launch daemons
cd ~/lachesis-2.0/scripts/lachesis
./setup-workers.sh 2 $SYSTEM

exit