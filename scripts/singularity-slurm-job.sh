#!/bin/bash

#SBATCH --mem=125G       	# Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-01:00     	# DD-HH:MM:SS
#SBATCH --constraint=broadwell 	# Request Broadwell processor
#SBATCH --nodes=1-1
#SBATCH --job-name=gosdt
#SBATCH --output=%x-%j.out

# Load dependencies
module load singularity/3.7

# set the error abort
set -euo pipefail

# set the variables
if [[ $# -eq 3 ]]; then
    GOSDT_DIR=$1
    CFG_FILE=$2
    RESULTS_DIR=$3
    ARRAY_OFFSET=0
else
    GOSDT_DIR=$1
    CFG_FILE=$2
    ARRAY_OFFSET=$3
    RESULTS_DIR=$4
fi

if [[ -v SLURM_ARRAY_TASK_ID ]]; then
    CONFIG_ID=$((ARRAY_OFFSET + SLURM_ARRAY_TASK_ID))
    echo "config id from slurm array id and offset: ${CONFIG_ID} = ${ARRAY_OFFSET} + ${SLURM_ARRAY_TASK_ID}"
else
    CONFIG_ID=$ARRAY_OFFSET
    echo "config id from supplied offset: ${CONFIG_ID} = ${ARRAY_OFFSET}"
fi


# the singularity image
IMAGE="/project/def-mseltzer/gosdt/gosdt-singuarity-image.sif"

# the temporary home directory
HOMEDIR=${RESULTS_DIR}/gosdt-homedir

# create temporary home directory locally to the machine
mkdir -p ${HOMEDIR}

# print the command
set -x

# execute the image, bind mount the directory and execute the wrapper
#
# Bind Mount: it maps a directory on the host (e.g., ${GOSDT_DIR}) to a directory within
#             the singularity image.
#             We use this to map the gosdt files into the image at /gosdt
#             Further we can bind-mount subdirectories, for example the results will be a bind
#             mount as well, i.e., /gosdt/results will be mapped to ${RESULTS_DIR} on the host
#
#             The exect script will CD into `/gosdt` when executed
#
# Command:    The command executed here is `bash <SCRIPT>`. The script to be executed must
#             exist within the image. Here we pass the script in the bind mount to bash to be
#             executed. Thus we need to set the path to `/gosdt`
#
# Note:       We explicitly mount a home directory with `-H`
singularity exec --cleanenv=true -H ${HOMEDIR} \
            --bind ${GOSDT_DIR}:/gosdt --bind ${RESULTS_DIR}:/gosdt/results \
            ${IMAGE}  bash /gosdt/scripts/singularity-exec.sh ${CFG_FILE} ${CONFIG_ID}
