#!/bin/bash

sudo apt update && sudo apt upgrade -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.10 -y
python3.10 --version
sudo apt install python3.10-dbg -y
sudo apt install python3.10-dev -y
sudo apt install python3.10-venv -y
sudo apt install python3.10-distutils -y
sudo apt install python3.10-lib2to3 -y
sudo apt install python3.10-gdbm -y
curl https://bootstrap.pypa.io/get-pip.py | sudo python3.10