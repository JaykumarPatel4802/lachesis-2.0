#!/bin/sh

cd /home/${USER}/openwhisk-3.0/ansible
ansible-playbook openwhisk.yml -e mode=clean