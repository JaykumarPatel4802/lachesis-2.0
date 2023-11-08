#!/bin/bash

cd ~
git clone https://github.com/RRZE-HPC/likwid.git
cd likwid
make
sudo make install
which likwid-powermeter # should see /usr/local/bin/likwid-powermeter
