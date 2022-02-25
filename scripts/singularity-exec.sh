#!/bin/bash

# stop if there was an error
set -euo pipefail

# set the variables
CFG_FILE=$1     # the configuration csv file
CFG_INDEX=$2    # the index into the csv file

# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------

# ASSUME THAT THINGS ARE ALREDY BUILT IN THE $HOME

# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

# move back to the gosdt directory
cd /gosdt

echo "config file: ${CFG_FILE}"
if [ ! -f ${CFG_FILE} ]; then
    echo "Configuration file does not exist."
    exit 1
fi

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

TIME_START=$(date "+%Y%m%d%H%M%S")
echo "start: ${TIME_START}"

# set the environment variables for the python script
export TB_ENV_GOSDT_VERSION=${GOSDT_VERSION}
export TB_ENV_TREE_BENCHMARK_VERSION=${TREEBENCHMARK_VERSION}

echo "python3 python/run.py csv ${CFG_FILE} ${CFG_INDEX}"
echo "-----------------------------------------------------------------"
python3 python/run.py csv ${CFG_FILE} ${CFG_INDEX}
echo "-----------------------------------------------------------------"

echo "end:"
date "+%Y-%m-%d-%H:%M:%S"

