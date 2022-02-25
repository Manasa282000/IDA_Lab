##!/bin/bash

# stop if there was an error
set -euo pipefail

# obtaining the submodules contents etc.
git submodule init
git submodule update

GOSDT_VERSION=$(cd gosdt && git rev-parse HEAD)
echo "gosdt version: ${GOSDT_VERSION}"

(cd gosdt && python3 setup.py build)
(cd gosdt && python3 setup.py install --user)

echo "gosdt python module built and installed"

DL85_VERSION=$(cd gosdt && git rev-parse HEAD)
echo "dl85 version: ${DL85_VERSION}"

(cd dl85 && python3 setup.py build)
(cd dl85 && python3 setup.py install --user)

echo "gosdt dl85 module built and installed"