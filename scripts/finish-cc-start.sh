#!/bin/bash

#################################################################################################
#
# Params:
#  -r <RESULTSDIR>       use this directory for the results
#
# Example usage:
#  to run using a pre-generated configuration file
#
#  $ bash singularity_start.sh -r results-v1
#
#  to generate the configuration file for running it:
#
#  $ base prepare-cc-start.sh -r results-v2 -g inputconfig.csv
#
################################################################################################

set -euo pipefail

echo "Compute Canada Finish Script (Singularity)"
echo "==============================================="

ROOT=$(git rev-parse --show-toplevel)
echo "repository root: ${ROOT}"

# switch to the root directory
cd ${ROOT}

# check if we have outstanding changes
echo "checking git for outstanding changes..."
if [[ $(git diff-index  HEAD --) != "" ]]; then
  echo "ERROR: the repository is not clean. Can't start a run."
  echo "please commit the outstanding changes"
  exit 1
fi
echo "OK. git status is clean."

DTNOW=$(date +%Y%m%d%H%M%S)
GOSDT_VERSION=$(cd gosdt && git rev-parse HEAD)
TREEBENCHMARK_VERSION=$(git rev-parse HEAD)

RESULTS_DIR=${ROOT}/results-${DTNOW}-${TREEBENCHMARK_VERSION:0:8}-${GOSDT_VERSION:0:8}
while getopts ":c:r:g:" opt; do
  case ${opt} in
    r )
      RESULTS_DIR=$(readlink -f $OPTARG)
      ;;
    \? )
      echo "Invalid option: $OPTARG" 1>&2
      ;;
    : )
      echo "Invalid option: $OPTARG requires an argument" 1>&2
      ;;
  esac
done

echo "results dir: ${RESULTS_DIR}"
echo "gosdt version: ${GOSDT_VERSION}"
echo "tree-benchmark version: ${TREEBENCHMARK_VERSION}"
echo "---------------------------------------"

# store the timestamp of the start
echo ${GOSDT_VERSION} > ${RESULTS_DIR}/version-gosdt.finish
echo ${TREEBENCHMARK_VERSION} > ${RESULTS_DIR}/version-tree-benchmark.finish

echo "---------------------------------------"

# set the slurm job file
JOB_FILE=${ROOT}/scripts/finish-cc-slurm-job.sh
echo "slurm job file: ${JOB_FILE}"

# create the slurm logs directory

# set the outfile to be within the slurmlogs directory
OUT_FILE=${RESULTS_DIR}/finish-slurmlog-%x-%a.out
echo "slurm output file: ${OUT_FILE}"

echo "---------------------------------------"

echo "sbatch -o ${OUT_FILE}  ${JOB_FILE} ${ROOT} ${RESULTS_DIR}"
sbatch -o ${OUT_FILE} ${JOB_FILE} ${ROOT} ${RESULTS_DIR}

echo "---------------------------------------"
