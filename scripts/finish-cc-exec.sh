#!/bin/bash

#################################################################################################
#
# Params:
#  - <RESULTSDIR>       use this file as
#
# Example usage:
#
#  $ bash singularity_finish.sh results-v2
#
################################################################################################

set -euo pipefail

NUM_WARNINGS=0

if [ "$#" == "0" ]; then
  echo "usage: bash singularity_finish.sh <RESULTSDIR>"
  exit 1
fi

# change directory if passed in
if [ "$#" == "2" ]; then
  cd $2
fi

# set the results directory
RESULTS_DIR=$1

echo "Compute Canada Finish Script (Singularity)"
echo "========================================================================="
echo ""

ROOT=$(git rev-parse --show-toplevel)
echo "[        ] repository root:   ${ROOT}"
echo "[        ] results directory: ${RESULTS_DIR}"
if [ ! -d ${RESULTS_DIR} ]; then
  echo -e "\e[1;31m[ ERROR  ]\e[0m results directory doe snot exists, or is not a directory"
  exit 1
fi

# get the current version of GOSDT and TREEBENCHMARK
pushd ${ROOT} > /dev/null
GOSDT_VERSION_REPO=$(cd gosdt && git rev-parse HEAD)
TREEBENCHMARK_VERSION_REPO=$(git rev-parse HEAD)

# check if we have outstanding changes
echo "[        ] checking git for outstanding changes..."
if [[ $(git diff-index  HEAD --) != "" ]]; then
  echo -e "\e[1;33m[  WARN  ]\e[0m repository has outstanding changes."
else
  echo -e "\e[1;32m[   OK   ]\e[0m git status is clean"
fi

popd > /dev/null

echo "[        ] checking gosdt versions"
GOSDT_VERSION=$(cat ${RESULTS_DIR}/version-gosdt)
if [ ${GOSDT_VERSION} != ${GOSDT_VERSION_REPO} ]; then
  echo -e "\e[1;33m[  WARN  ]\e[0m gosdt version of the repository differs from the results directory."
  echo -e "\e[1;33m[  WARN  ]\e[0m gosdt version (repo):             ${GOSDT_VERSION_REPO}"
  echo -e "\e[1;33m[  WARN  ]\e[0m gosdt version (results):          ${GOSDT_VERSION}"
  NUM_WARNINGS=$((NUM_WARNINGS + 1))
else
  echo -e "\e[1;32m[   OK   ]\e[0m gosdt versions match"
fi

echo "[        ] checking treebenchmark versions"
TREEBENCHMARK_VERSION=$(cat ${RESULTS_DIR}/version-tree-benchmark)
if [ ${TREEBENCHMARK_VERSION} != ${TREEBENCHMARK_VERSION_REPO} ]; then
  echo -e "\e[1;33m[  WARN  ]\e[0m tree-benchmark version of the repository differs from the results directory."
  echo -e "\e[1;33m[  WARN  ]\e[0m tree-benchmark version (repo):    ${TREEBENCHMARK_VERSION}"
  echo -e "\e[1;33m[  WARN  ]\e[0m tree-benchmark version (results): ${TREEBENCHMARK_VERSION_REPO}"
  NUM_WARNINGS=$((NUM_WARNINGS + 1))
else
  echo -e "\e[1;32m[   OK   ]\e[0m tree-benchmark versions match"
fi

echo "[        ] getting the number of configurations"
if [ ! -f ${RESULTS_DIR}/configs.csv ]; then
  echo -e "\e[1;31m[ ERROR  ]\e[0m configuration file `${RESULTS_DIR}/configs.csv` does not exist."
  exit 1
fi

NUM_LINES=$(wc -l ${RESULTS_DIR}/configs.csv | cut -f 1 -d ' ')
NUM_CONFIGS_EXPECTED=$((NUM_LINES - 1))
NUM_CONFIGS=$(ls ${RESULTS_DIR}/configmap | wc -l | cut -f 1 -d ' ')

echo -e "\e[1;32m[   OK   ]\e[0m expected configurations: ${NUM_CONFIGS_EXPECTED}"

# =========================================================================================
# Configurations
# =========================================================================================

TIMESTAMP=$(cat ${RESULTS_DIR}/timestamp)

PROCESSED_DIR_NAME=tb-results-${TIMESTAMP}-${TREEBENCHMARK_VERSION:0:8}
PROCESSED_DIR=${RESULTS_DIR}/${PROCESSED_DIR_NAME}

# remove the output directory
if [ -d ${PROCESSED_DIR} ]; then
  rm -rf ${PROCESSED_DIR}
fi

# create the processed dir
mkdir -p ${PROCESSED_DIR}
cp ${RESULTS_DIR}/configs.csv ${PROCESSED_DIR}

# =========================================================================================
# The configuration map
# =========================================================================================



CONFIG_MAP_FILE=${PROCESSED_DIR}/configmap.json

echo "[        ] collecting configmap..."
echo "[" > ${CONFIG_MAP_FILE}
NUM_NOT_FOUND=0
for cf in `seq 0 $NUM_CONFIGS_EXPECTED`; do
    echo -n "\"" >> ${CONFIG_MAP_FILE}
    if [ -f ${RESULTS_DIR}/configmap/${cf} ]; then
      cat ${RESULTS_DIR}/configmap/${cf} >> ${CONFIG_MAP_FILE}
    else
      echo -n "None" >> ${CONFIG_MAP_FILE}
      NUM_WARNINGS=$((NUM_WARNINGS + 1))
      NUM_NOT_FOUND=$((NUM_NOT_FOUND + 1))
    fi
    echo "\"," >> ${CONFIG_MAP_FILE}
done
echo "\"LAST\"" >> ${CONFIG_MAP_FILE}
echo "]" >> ${CONFIG_MAP_FILE}


# =========================================================================================
# Now look through the results
# =========================================================================================

# the failed runs file
RUN_STATUS_FILE=${PROCESSED_DIR}/runstatus.csv
echo "idx,runstatus" > ${RUN_STATUS_FILE}

# the combined csv file
ALL_RESULTS=${PROCESSED_DIR}/allresults.csv

# get one results file
RESFILE=$(find ${RESULTS_DIR}/runlogs -name results.csv | head -n 1) || true

# get the header
echo -n "runstatus," > ${ALL_RESULTS}
head -n 1 ${RESFILE} >> ${ALL_RESULTS}

NUM_SLURM_LOG_MISSING=0
NUM_SLURM_TIMEOUT=0
NUM_SLURM_OOM=0
NUM_NO_RUNLOGS=0
NUM_DATASET_FAILURES=0
NUM_RUN_FAILURES=0
NUM_ALG_TIMEOUT=0
NUM_SKIPPED=0

SL_SUC_DIR=${PROCESSED_DIR}/success
SL_TIMEOUT_DIR=${PROCESSED_DIR}/fail_slurm_timeout
SL_OOM_DIR=${PROCESSED_DIR}/fail_slurm_oom
SL_UNKNOWN_DIR=${PROCESSED_DIR}/fail_slurm_unknown
SL_ALG_FAIL_DIR=${PROCESSED_DIR}/fail_algo
SL_DATASET_FAIL_DIR=${PROCESSED_DIR}/fail_dataset
SL_UNKNONW_ERR=${PROCESSED_DIR}/fail_run_unknown

mkdir -p ${SL_SUC_DIR}
mkdir -p ${SL_TIMEOUT_DIR}
mkdir -p ${SL_OOM_DIR}
mkdir -p ${SL_UNKNOWN_DIR}
mkdir -p ${SL_ALG_FAIL_DIR}
mkdir -p ${SL_DATASET_FAIL_DIR}
mkdir -p ${SL_UNKNONW_ERR}

echo "[        ] processing results"
CFG_IDX_END=$((NUM_CONFIGS_EXPECTED - 1))
for cf in `seq 0 $CFG_IDX_END`; do

  # check if the slurm log exists
  SLURMLOG=${RESULTS_DIR}/slurmlogs/slurmlog-gosdt-${cf}.out
  if [ ! -f ${SLURMLOG} ]; then
    echo "${cf},FAIL_SLURM_LOG_MISSING" >> ${RUN_STATUS_FILE}
    NUM_WARNINGS=$((NUM_WARNINGS + 1))
    NUM_SLURM_LOG_MISSING=$((NUM_SLURM_LOG_MISSING + 1))
    continue
  fi

  if grep -i 'already run' ${SLURMLOG} > /dev/null; then
    NUM_SKIPPED=$((NUM_SKIPPED + 1))
  fi

  if [ ! -f ${RESULTS_DIR}/configmap/${cf} ]; then
    echo "${cf},FAIL_NO_CONFIGMAP" >> ${RUN_STATUS_FILE}
    continue
  fi

  # get the hash
  HASH=$(cat ${RESULTS_DIR}/configmap/${cf})
  RUNLOG_DIR=${RESULTS_DIR}/runlogs/${HASH:0:2}/${HASH}

  if [ ! -d ${RUNLOG_DIR} ]; then
    echo "${cf},FAIL_NO_RUNLOG_DIRECTORY" >> ${RUN_STATUS_FILE}
    NUM_NO_RUNLOGS=$((NUM_NO_RUNLOGS + 1))
    continue
  fi

  CONFIG_FILE=${RUNLOG_DIR}/config.json
  TREE_FILE=${RUNLOG_DIR}/tree.json

  STATUS="SUCCESS"

  # check if the slurm log indicates an error
  if grep 'slurmstepd: error' ${SLURMLOG} > /dev/null; then
    NUM_WARNINGS=$((NUM_WARNINGS + 1))
    if grep 'CANCELLED' ${SLURMLOG} > /dev/null; then
      NUM_SLURM_TIMEOUT=$((NUM_SLURM_TIMEOUT + 1))
      STATUS="FAIL_SLURM_TIMEOUT"
      cp ${SLURMLOG} ${SL_TIMEOUT_DIR}/${HASH}-run.log
      cp ${CONFIG_FILE} ${SL_TIMEOUT_DIR}/${HASH}-config.json
    elif grep 'oom-kill' ${SLURMLOG} > /dev/null; then
      NUM_SLURM_OOM=$((NUM_SLURM_OOM + 1))
      STATUS="FAIL_SLURM_OUT_OF_MEMORY"
      cp ${SLURMLOG} ${SL_OOM_DIR}/${HASH}-run.log
      cp ${CONFIG_FILE} ${SL_OOM_DIR}/${HASH}-config.json
    else
      echo -e "\e[1;33m[  WARN  ]\e[0m config ${cf} does had an error. (SLURM LOG)"
      grep 'slurmstepd: error' ${SLURMLOG}
      NUM_WARNINGS=$((NUM_WARNINGS + 1))
      STATUS="UNKNOWN_SLURM_ERROR"
      cp ${SLURMLOG} ${SL_UNKNOWN_DIR}/${HASH}-run.log
      cp ${CONFIG_FILE} ${SL_UNKNOWN_DIR}/${HASH}-config.json
    fi
  fi

  # if the results.csv exists we're good
  if [ -f ${RUNLOG_DIR}/results.csv ]; then
    if grep -i 'timeout' ${SLURMLOG} > /dev/null; then
      STATUS="ALGORITHM_TIMEOUT"
      NUM_ALG_TIMEOUT=$((NUM_ALG_TIMEOUT + 1))
    fi

    echo "${cf},${STATUS}" >> ${RUN_STATUS_FILE}
    echo -n "${STATUS}," >> ${ALL_RESULTS}
    tail -n 1 ${RUNLOG_DIR}/results.csv >> ${ALL_RESULTS}
    cp ${SLURMLOG} ${SL_SUC_DIR}/${HASH}-run.log
    cp ${TREE_FILE} ${SL_SUC_DIR}/${HASH}-tree.json
    cp ${CONFIG_FILE} ${SL_SUC_DIR}/${HASH}-config.json
    continue
  fi

  # if the errors.csv exists this is bad
  if [ -f ${RUNLOG_DIR}/error.csv ]; then
    echo "${cf},${STATUS}" >> ${RUN_STATUS_FILE}
    echo -n "${STATUS}," >> ${ALL_RESULTS}
    tail -n 1 ${RUNLOG_DIR}/error.csv >> ${ALL_RESULTS}
    NUM_WARNINGS=$((NUM_WARNINGS + 1))
    continue
  fi

  # if the results.tmp exists, there was some issue
  if [ -f ${RUNLOG_DIR}/results.tmp ]; then
    if grep 'RUN_FAILURE' ${RUNLOG_DIR}/results.tmp > /dev/null; then
      NUM_RUN_FAILURES=$((NUM_RUN_FAILURES + 1))
      STATUS="FAIL_ALGORITHM_RUN"
      cp ${SLURMLOG} ${SL_ALG_FAIL_DIR}/${HASH}-run.log
      cp ${CONFIG_FILE} ${SL_ALG_FAIL_DIR}/${HASH}-config.json
    elif grep 'DATASET_FAILURE' ${RUNLOG_DIR}/results.tmp > /dev/null; then
      NUM_DATASET_FAILURES=$((NUM_DATASET_FAILURES + 1))
      STATUS="FAIL_DATASET_LOADING"
      cp ${SLURMLOG} ${SL_DATASET_FAIL_DIR}/${HASH}-run.log
      cp ${CONFIG_FILE} ${SL_DATASET_FAIL_DIR}/${HASH}-config.json
    else
      STATUS="UNKNOWN_RUN_ERROR"
      cp ${SLURMLOG} ${SL_UNKNONW_ERR}/${HASH}-run.log
      cp ${CONFIG_FILE} ${SL_UNKNONW_ERR}/${HASH}-config.json
    fi
    echo -n "${STATUS}," >> ${ALL_RESULTS}
    tail -n 1 ${RUNLOG_DIR}/results.tmp >> ${ALL_RESULTS}
    echo "${cf},${STATUS}" >> ${RUN_STATUS_FILE}
    NUM_WARNINGS=$((NUM_WARNINGS + 1))
    continue
  fi

  echo -e "\e[1;33m[  WARN  ]\e[0m unknown run outcome ${cf}"
  NUM_WARNINGS=$((NUM_WARNINGS + 1))
done

echo -e "\e[1;32m[   OK   ]\e[0m resultes processed"
echo "[        ] status is as follows:"

if [ ${NUM_NOT_FOUND} != 0 ]; then
  echo -e "\e[1;33m[  WARN  ]\e[0m configmaps not found: ${NUM_NOT_FOUND}"
else
  echo -e "\e[1;32m[   OK   ]\e[0m all expected config maps entries found"
fi

if [ ${NUM_SLURM_LOG_MISSING} != 0 ]; then
  echo -e "\e[1;33m[  WARN  ]\e[0m slurm logs missing: ${NUM_SLURM_LOG_MISSING}"
else
  echo -e "\e[1;32m[   OK   ]\e[0m all expected slurm logs found"
fi

if [ ${NUM_SLURM_TIMEOUT} != 0 ]; then
  echo -e "\e[1;33m[  WARN  ]\e[0m slurm logs indicating timeouts: ${NUM_SLURM_TIMEOUT}"
else
  echo -e "\e[1;32m[   OK   ]\e[0m no slurm timeouts found"
fi

if [ ${NUM_SLURM_OOM} != 0 ]; then
  echo -e "\e[1;33m[  WARN  ]\e[0m slurm logs indicating out-of-memory: ${NUM_SLURM_OOM}"
else
  echo -e "\e[1;32m[   OK   ]\e[0m no slurm out-of-memory conditions detected"
fi

if [ ${NUM_NO_RUNLOGS} != 0 ]; then
  echo -e "\e[1;33m[  WARN  ]\e[0m number of missing runlogs: ${NUM_SLURM_OOM}"
else
  echo -e "\e[1;32m[   OK   ]\e[0m all expected runlog directories found"
fi

if [ ${NUM_DATASET_FAILURES} != 0 ]; then
  echo -e "\e[1;33m[  WARN  ]\e[0m number of dataset failures: ${NUM_DATASET_FAILURES}"
else
  echo -e "\e[1;32m[   OK   ]\e[0m no dataset-related failures detected"
fi

if [ ${NUM_RUN_FAILURES} != 0 ]; then
  echo -e "\e[1;33m[  WARN  ]\e[0m number of run failures: ${NUM_RUN_FAILURES}"
else
  echo -e "\e[1;32m[   OK   ]\e[0m no run failure detected"
fi

if [ ${NUM_SKIPPED} != 0 ]; then
  echo -e "\e[1;33m[  WARN  ]\e[0m number of skipped runs: ${NUM_SKIPPED}"
else
  echo -e "\e[1;32m[   OK   ]\e[0m no skipped runs"
fi

if [ ${NUM_ALG_TIMEOUT} != 0 ]; then
  echo -e "\e[1;33m[  WARN  ]\e[0m number of algorithm timeouts: ${NUM_ALG_TIMEOUT}"
else
  echo -e "\e[1;32m[   OK   ]\e[0m no algorithm timeouts detected"
fi

echo "[        ] combining CSVs"



echo -e "\e[1;32m[   OK   ]\e[0m resultes processed"




echo "[        ] compressing results..."
pushd ${RESULTS_DIR} > /dev/null
tar -czf ${PROCESSED_DIR_NAME}.tar.gz ${PROCESSED_DIR_NAME}
popd > /dev/null
echo -e "\e[1;32m[   OK   ]\e[0m all complete"
echo -e "\e[1;32m[   OK   ]\e[0m results archive file: ${RESULTS_DIR}/${PROCESSED_DIR_NAME}.tar.gz"
