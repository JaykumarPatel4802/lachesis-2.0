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
sudo -H pip uninstall --yes ansible
sudo apt remove -y ansible && sudo apt purge -y ansible && sudo apt autoremove
sudo -H pip uninstall --yes ansible-base ansible-core
sudo -H pip install ansible==4.1.0
sudo -H pip install jinja2==3.0.1
sudo apt-get install -y python3-testresources
sudo -H pip install --upgrade setuptools
sudo apt-get install -y default-jre
# sudo pip install --upgrade requests