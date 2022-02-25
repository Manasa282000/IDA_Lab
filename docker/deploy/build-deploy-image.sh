#!/bin/bash

set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
echo "repository root: ${ROOT}"

# cd to the root directory
pushd ${ROOT} > /dev/null

# initialize the submodules
git submodule init
git submodule update

# make sure we don't have any outstanding changes
(cd ${ROOT}/gosdt && git reset --hard)
(cd ${ROOT}/dl85 && git reset --hard)

# the docker file to build
DOCKERFILE=docker/deploy/Dockerfile

# IMAGE
DOCKERIMAGE=gosdt

# build the image
echo "BUILDING IMAGE"
docker build -t ${DOCKERIMAGE} -f ${DOCKERFILE} .

# restore directory
popd > /dev/null