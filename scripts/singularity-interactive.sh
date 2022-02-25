#!/bin/bash

#SBATCH --mem=128G       	# Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-01:00     	# DD-HH:MM:SS
#SBATCH --constraint=broadwell 	# Request Broadwell processor
#SBATCH --nodes=1-1
#SBATCH --job-name=gosdt
#SBATCH --output=%x-%j.out

# Load dependencies
module load singularity/3.7

set -seuo pipefail

# set the variables
GOSDT_DIR=$1
CFG_FILE=$2
RESULTS_DIR=$3

# the singularity image
IMAGE="/project/def-mseltzer/gosdt/gosdt-singuarity-image.sif"

echo "singularity exec --bind ${GOSDT_DIR}:/gosdt --bind ${RESULTS_DIR}:/gosdt/results \
            ${IMAGE}  bash /gosdt/scripts/singularity-exec.sh $CFG_FILE ${SLURM_ARRAY_TASK_ID}"

# create temporary home directory
mdkir -p /tmp/gosdt-home

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
singularity shell --cleanenv=true -H /tmp/gosdt-home \
            --bind ${GOSDT_DIR}:/gosdt --bind ${RESULTS_DIR}:/gosdt/results \
            ${IMAGE}