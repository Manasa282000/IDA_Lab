#!/bin/bash

set -euo pipefail

echo "================================================"
echo "GOSDT DOCKER CONTAINER"
echo "================================================"

# this is the docker entrypoint
cd /gosdt || exit

if [ $# -eq 0 ]; then
    echo "starting shell..."
    exec "/bin/bash"
    exit 0
else
    echo "executing '$*'"
    exec "$*"
fi