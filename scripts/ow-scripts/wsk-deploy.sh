#!/bin/sh

# Setup CouchDB and hosts
cd /home/${USER}/openwhisk-3.0/ansible
export OW_DB=CouchDB
export OPENWHISK_TMP_DIR=/home/${USER}/openwhisk-tmp-dir
ENVIRONMENT=local
ansible-playbook -i environments/$ENVIRONMENT setup.yml
ansible-playbook -i environments/$ENVIRONMENT prereq.yml

# Start (almost) everything
ansible-playbook -i environments/$ENVIRONMENT couchdb.yml
ansible-playbook -i environments/$ENVIRONMENT initdb.yml
ansible-playbook -i environments/$ENVIRONMENT wipe.yml
ansible-playbook -i environments/$ENVIRONMENT openwhisk.yml

# Install catalog of public packages and actions
ansible-playbook -i environments/$ENVIRONMENT postdeploy.yml

# Enable the use of the API gateway
ansible-playbook -i environments/$ENVIRONMENT apigateway.yml
ansible-playbook -i environments/$ENVIRONMENT routemgmt.yml

# Error list
# 1. couchdb.yml was throwing error below:
# Failed to import the required Python library (Docker SDK for Python: docker
# above 5.0.0 (Python >= 3.6) or docker before 5.0.0 (Python 2.7) or docker-py
# (Python 2.6)) on ubuntu's Python /usr/bin/python. Please read the module
# documentation and install it in the appropriate location. If the required
# library is installed, but Ansible is using the wrong Python interpreter, please
# consult the documentation on ansible_python_interpreter, for example via `pip
# install docker` (Python >= 3.6) or `pip install docker==4.4.4` (Python 2.7) or
# `pip install docker-py` (Python 2.6). The error was: No module named
# requests.exceptions

# Solution: Change ansible.cfg to use the /usr/bin/python3 interpreter

# 2. postdeploy.yml throwing error below: 
# An exception occurred during task execution. To see the full traceback, use -vvv. 
# The error was: ansible.errors.AnsibleUndefinedVariable: {'protocol': "{{ scheduler_protocol | default('http') }}", 'dir': {'become':
# Solution: ignore, not needed

# https://github.com/docker/docker-py/issues/3113
