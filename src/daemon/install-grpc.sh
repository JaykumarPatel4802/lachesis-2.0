#!/bin/bash

cd ~
curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v3.11.4/protoc-3.11.4-linux-x86_64.zip
unzip protoc-3.11.4-linux-x86_64.zip -d protoc3

# Move the binary to directory which is PATH
sudo mv protoc3/bin/* /usr/local/bin/

sudo mv protoc3/include/* /usr/local/include/

# Change owner
sudo chown $USER /usr/local/bin/protoc
sudo chown -R $USER /usr/local/include/google

# Test if it works
protoc --version

# Install grpcio and grpcio-tools
pip3 install grpcio grpcio-tools