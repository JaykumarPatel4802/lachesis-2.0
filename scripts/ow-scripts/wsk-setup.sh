#!/bin/sh

# Housekeeping with the new instance
sudo apt-get update
sudo apt-get -y upgrade

# Install required dependencies
sudo apt-get install -y nodejs-dev node-gyp libssl1.0-dev
sudo apt-get install -y libnode-dev # cascade intel processor
sudo apt-get install -y npm
sudo apt-get install -y python-pip
sudo apt-get install -y python3-pip # cascade intel processor
sudo -H pip install --upgrade pip
sudo -H pip uninstall ansible
sudo apt remove ansible && sudo apt purge ansible && sudo apt autoremove
sudo -H pip uninstall ansible-base ansible-core
sudo -H pip install ansible==4.1.0
sudo -H pip install jinja2==3.0.1
sudo apt-get install -y python3-testresources
sudo -H pip install --upgrade setuptools
sudo apt-get install -y default-jre
# sudo pip install --upgrade requests

# Build openwhisk
cd ~/
git clone https://github.com/neerajas-group/openwhisk-3.0.git
cd openwhisk-3.0
./tools/ubuntu-setup/all.sh
sudo apt-get install -y npm
./gradlew distDocker
