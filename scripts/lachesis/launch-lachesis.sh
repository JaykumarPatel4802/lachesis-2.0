#!/bin/bash

SYSTEM=$1

# 1. Clear prediction table in lachesis database -- this will remove learned online models also
cd ~/lachesis/src/datastore
python3 datastore-lachesis.py
echo "Cleared prediction tables and removed learned online models"

# 2. Setup lachesis daemons
cd ~/lachesis/scripts/lachesis
./setup-workers.sh 0 $SYSTEM

# 3. Redeploy OpenWhisk controller - remove any state
cd ~/lachesis/scripts/ow-scripts
./wsk-redeploy-component.sh controller

# 4. Launch daemons
cd ~/lachesis/scripts/lachesis
./setup-workers.sh 2 $SYSTEM

exit