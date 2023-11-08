#!/bin/bash

for ((no=0; no<60; no++)); do
    sudo turbostat --quiet --show CPU,CPU%c6 --interval 5 --num_iterations 1 >> ~/lachesis/data/energy-study/raw-cstate-data.txt
    echo " " >> ~/lachesis/data/energy-study/raw-cstate-data.txt
done