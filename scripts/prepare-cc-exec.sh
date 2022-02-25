#!/bin/bash

# stop if there was an error
set -euo pipefail

# go into the bind-mounted directory
cd /gosdt

echo "preparing for runs"

GOSDT_VERSION=$(cd gosdt && git rev-parse HEAD)
echo "gosdt version: ${GOSDT_VERSION}"
git rev-parse HEAD:gosdt

TREEBENCHMARK_VERSION=$(git rev-parse HEAD)
echo "tree-benchmark version: ${TREEBENCHMARK_VERSION}"

# check if we have outstanding changes
if [[ $(git diff-index  HEAD --) != "" ]]; then
  echo "============================================================="
  echo "!!! WARNING !!!  THE REPOSITORY IS NOT CLEAN  !!! WARNING !!!"
  echo "============================================================="
fi

# set the versions
echo ${GOSDT_VERSION} > ${HOME}/version-gosdt
echo ${TREEBENCHMARK_VERSION} > ${HOME}/version-tree-benchmark

# ---------------------------------------------------------------------------
# Building GOSDT
# ---------------------------------------------------------------------------

echo "-----------------------------------------------------------------------"
echo "building pygosdt..."

# go into the gosdt module
cd /gosdt/gosdt

# clean the directory
python3 setup.py clean

# build it in a machine-local directory
python3 setup.py build -q -j `nproc`  &> ${HOME}/gosdtbuild.log

# install the gosdt python module in the mounted home directory
python3 setup.py install --skip-build  --user

echo "pygosdt installed"

# ---------------------------------------------------------------------------
# Building DL8.5
# ---------------------------------------------------------------------------

echo "-----------------------------------------------------------------------"
echo "building pydl8.5..."

# go into the bind-mounted directory
cd /gosdt/dl85

# clean the directory
python3 setup.py clean

echo "building pydl85..."
python3 setup.py build -q -j `nproc` &> ${HOME}/dl85build.log

# install the dl85 python module in the home directory
echo "installig pydl85"
python3 setup.py install --skip-build  --user

echo "pydl85 installed"

# ---------------------------------------------------------------------------
# Binarize Datasets
# ---------------------------------------------------------------------------

echo "-----------------------------------------------------------------------"
echo "binarizing datasets... (TODO)"

cd /gosdt/datasets

# create the target directory if not exists
mkdir -p /gosdt/datasets/binarized_datasets

# install pyarrow for the feather dependency
pip3 install pyarrow --user

for d in broward_general_2y  compas  coupon_carryout  coupon_rest20  fico  netherlands  spiral; do
  echo "skipping binarization of datasets"
  python3 scripts/parallel_binarize.py $d
done

echo "-----------------------------------------------------------------------"
echo "end:"


