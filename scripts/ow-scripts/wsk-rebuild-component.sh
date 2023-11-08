#!/bin/sh

if [ "$#" -lt 1 ]; then
  echo "Usage: wsk-rebuild-component COMPONENT [TAG]"
  exit 1
fi

if [ "$#" -lt 2 ]; then
  echo "Using default tag \`latest\`"
  set -- "$1" "latest"
fi

cd /home/${USER}/openwhisk-3.0
./gradlew :core:$1:distDocker -PdockerImageTag=$2
cd /home/${USER}/openwhisk-3.0