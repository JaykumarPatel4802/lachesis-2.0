#!/bin/sh

if [ "$#" -lt 1 ]; then
  echo "Usage: wsk-component-redeploy COMPONENT [TAG]"
  exit 1
fi

if [ "$#" -lt 2 ]; then
  echo "Using default tag \`latest\`"
  set -- "$1" "latest"
fi

export OPENWHISK_TMP_DIR=/home/${USER}/openwhisk-tmp-dir
ENVIRONMENT=local

cd /home/${USER}/openwhisk-3.0
./gradlew :core:$1:distDocker -PdockerImageTag=$2

cd ansible
ansible-playbook -i environments/$ENVIRONMENT $1.yml -e docker_image_tag=$2