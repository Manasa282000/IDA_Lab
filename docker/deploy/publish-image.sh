#!/bin/bash

set -euo pipefail

# the local docker image
DOCKERIMAGE=gosdt

# the target docker image (pushed)
TARGET_IMAGE=achreto/gosdt

# tag the image
docker tag ${DOCKERIMAGE} ${TARGET_IMAGE}

# depoloy the image
echo "PUSH IMAGE"
docker push ${TARGET_IMAGE}