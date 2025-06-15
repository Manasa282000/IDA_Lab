#!/bin/bash

# Stop if there was an error
set -euo pipefail

# Initialize and update git submodules
git submodule init
git submodule update

# Get the DL8.5 version
DL85_VERSION=$(cd dl85 && git rev-parse HEAD)
echo "dl85 version: ${DL85_VERSION}"

# Build and install the DL8.5 Python module
(cd dl85 && python3 setup.py build)
(cd dl85 && python3 setup.py install --user)

echo "dl85 python module built and installed"