#!/bin/bash

# Download Go (https://go.dev/doc/install)
cd $HOME
sudo wget https://go.dev/dl/go1.19.2.linux-amd64.tar.gz -P /usr/local
sudo rm -rf /usr/local/go
cd /usr/local/
sudo tar -C /usr/local -xzf go1.19.2.linux-amd64.tar.gz
echo "export PATH=$PATH:/usr/local/go/bin" >> ~/.bashrc
source $HOME/.bashrc
go version # (should see: go version go1.19.2 linux/amd64)

# Pull and build OpenWhisk CLI
cd /home/${USER}
git clone https://github.com/neerajas-group/openwhisk-cli
cd openwhisk-cli
go get -u github.com/jteeuwen/go-bindata/...
sudo apt install go-bindata
go-bindata -pkg wski18n -o wski18n/i18n_resources.go wski18n/resources
go build -o wsk

# Add OpenWhisk client to $PATH
echo "export PATH=$PATH:$HOME/openwhisk-cli" >> ~/.bashrc
source $HOME/.bashrc

# Bypass certificate checking
echo "alias wsk='wsk -i'" >> ~/.bashrc
source $HOME/.bashrc

# Setup OpenWhisk client properties
host_ip=$(getent hosts `hostname` | awk '{ print $1 }')
port=443
wsk property set --apihost=$host_ip:$port
wsk property set --auth `cat $HOME/openwhisk-3.0/ansible/files/auth.guest`
