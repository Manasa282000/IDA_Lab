#!/bin/bash

#SBATCH --mem=400G
#SBATCH --time=00-01:00:00
#SBATCH --job-name=generate-netherlands-dataset
#SBATCH --output=%x-%j.out

set -euo pipefail

# Load dependencies
module load python/3.8

echo "Dependencies loaded"

source ../my_env/bin/activate
echo "finished environement activation"

mkdir -p binarized_datasets/netherlands

python3 scripts/binarize.py netherlands

echo "dataset generated"

deactivate
