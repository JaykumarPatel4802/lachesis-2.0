#!/bin/bash

# 1. Set up the repository

# Update existing list of packages
sudo apt update

# Install a few prerequisite packages which let apt use packages over HTTPS
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker’s official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Add Docker repository to APT sources
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 2. Install Docker Engine

# Set up default umask for GPG key correctly so that public key file for the repo can be detected
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Install latest version of docker engine, containerd, and docker compose
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# See that docker process is up and running
sudo systemctl status docker

# Add username to docker group so I don't have to type sudo everytime
sudo usermod -aG docker ${USER}

# Log in and out the server
# Confirm that I am added to the docker group (should see docker appear after typing)
groups

# Verify that the Docker Engine is installed correctly by running hello-world image
sudo service docker start
sudo docker run hello-world