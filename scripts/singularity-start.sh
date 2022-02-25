#!/bin/bash

#################################################################################################
#
# Params:
#  -c <CONFIGFILE>       use the following configuration file
#  -g <CONFIGTEMPLATE>   use this file as a template to generate the configuration file
#  -r <RESULTSDIR>       use this file as
#
# Example usage:
#  to run using a pre-generated configuration file
#
#  $ bash singularity_start.sh -r results-v1 -c myconfig.csv
#
#  to generate the configuration file for running it:
#
#  $ base singularity_start.sh -r results-v2 -g inputconfig.csv
#
################################################################################################

set -euo pipefail

echo "Compute Canada Run Script (Singularity)"
echo "======================================="

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

CONFIG_FILE=${DTNOW}-runconfig.csv
RESULTS_DIR=${ROOT}/results-${DTNOW}-${TREEBENCHMARK_VERSION:0:8}-${GOSDT_VERSION:0:8}
GENERATE_CONFIG_FILE=${ROOT}/configtemplate.csv
SKIP_GEN=0
while getopts ":c:r:g:" opt; do
  case ${opt} in
    c )
      CONFIG_FILE=$OPTARG
      SKIP_GEN=1
      ;;
    r )
      RESULTS_DIR=$(readlink -f $OPTARG)
      ;;
    g )
      GENERATE_CONFIG_FILE=$OPTARG
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

# create the results directory
if [ ! -d ${RESULTS_DIR} ]; then
    mkdir -p ${RESULTS_DIR}
fi

if [ ! -d ${RESULTS_DIR}/gosdt-homedir ]; then
    echo "home dir missing. run the following command first:"
    echo "bash scripts/prepare-cc-start.sh -r ${RESULTS_DIR}"
    exit 1
fi

if [ ! -d ${ROOT}/datasets/binarized_datasets ]; then
  echo "binarized dataset directory missing. run the following command first:"
  echo "bash scripts/prepare-cc-start.sh -r ${RESULTS_DIR}"
  echo "or copy datasets/pre_binarized_datasets to datasets/binarized_datasets"
  exit 1
fi

# store the timestamp of the start
echo ${DTNOW} > ${RESULTS_DIR}/timestamp
echo ${GOSDT_VERSION} > ${RESULTS_DIR}/version-gosdt
echo ${TREEBENCHMARK_VERSION} > ${RESULTS_DIR}/version-tree-benchmark

# TODO: make sure the datasets are ready here....

if [ "${SKIP_GEN}" == "0" ]; then
  echo "generating configuration file from ${GENERATE_CONFIG_FILE}"

  # TODO: create the configuration file here!
fi

# set the configuration file
echo "configuration file: ${CONFIG_FILE}"

if [ ! -f ${CONFIG_FILE} ]; then
  echo "Configuration file does not exist."
  exit 1
fi

# get the number of configuration lines
NUM_LINES=$(wc -l ${CONFIG_FILE} | cut -f 1 -d ' ')
NUM_CONFIGS=$((NUM_LINES - 1))
CFG_ID_END=$((NUM_CONFIGS - 1))

# copy the configuration file into the results directory
cp ${CONFIG_FILE} ${RESULTS_DIR}/configs.csv

# that will show up here in the image
CFG_FILE="results/configs.csv"

echo "---------------------------------------"

# set the slurm job file
JOB_FILE=${ROOT}/scripts/singularity-slurm-job.sh
echo "slurm job file: ${JOB_FILE}"

# create the slurm logs directory
SLURMLOGS_DIR=${RESULTS_DIR}/slurmlogs
mkdir -p ${SLURMLOGS_DIR}
# set the outfile to be within the slurmlogs directory
OUT_FILE=${SLURMLOGS_DIR}/slurmlog-%x-%a.out
echo "slurm output file: ${OUT_FILE}"

# the configuration map CFG_ID -> hash
mkdir -p ${RESULTS_DIR}/configmap

# the runlogs directory
mkdir -p ${RESULTS_DIR}/runlogs

echo "---------------------------------------"
echo "enqueue ${NUM_CONFIGS} jobs..."

MAX_ARRAY_JOBS=$(scontrol show config | grep MaxArraySize | cut -d= -f2)
ARRAY_START=0

ARRAY_END=$((MAX_ARRAY_JOBS - 1))
while [[ $NUM_CONFIGS -gt $MAX_ARRAY_JOBS ]]; do
  # calculate the end of the array

  # enqueue the batch for the current values
  echo "sbatch --array=0-${ARRAY_END}  -o ${OUT_FILE}  ${JOB_FILE} ${ROOT} ${CFG_FILE} ${ARRAY_START} ${RESULTS_DIR}"
  sbatch --array=0-${ARRAY_END} -o ${OUT_FILE} ${JOB_FILE} ${ROOT} ${CFG_FILE} ${ARRAY_START} ${RESULTS_DIR}

  # calculate the new array start
  ARRAY_START=$((ARRAY_START + MAX_ARRAY_JOBS))
  # calculate the remaining configurations
  NUM_CONFIGS=$((NUM_CONFIGS - MAX_ARRAY_JOBS))

  # wait a little bit, 2 seconds
  sleep 2
done

ARRAY_END=$((NUM_CONFIGS - 1))

# enqueue the remainder
echo "sbatch --array=0-${ARRAY_END}  -o ${OUT_FILE}  ${JOB_FILE} ${ROOT} ${CFG_FILE} ${ARRAY_START} ${RESULTS_DIR}"
sbatch --array=0-${ARRAY_END} -o ${OUT_FILE} ${JOB_FILE} ${ROOT} ${CFG_FILE} ${ARRAY_START} ${RESULTS_DIR}