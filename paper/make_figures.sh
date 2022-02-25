#!/bin/bash

# stop if there was an error
set -euo pipefail

# depends
# sudo apt install python3-openpyxl

# setting root directory of the git folder
ROOTDIR=$(git rev-parse --show-toplevel)

cd ${ROOTDIR}
bash ./scripts/convert_results.sh

cd ${ROOTDIR}/Threshold-Initial-Setup
bash convert_results.sh

cd ${ROOTDIR}
python3 ./scripts/merge_excel.py results/all_data.xlsx results/threshold_data.xlsx results/combined_data.xlsx

cd ${ROOTDIR}/paper

mkdir -p figures

echo "###############################################################################"
echo "Creating Paper Figures"
echo "###############################################################################"

echo "==============================================================================="
echo "Figure 2"
echo "==============================================================================="
python3 figure2.py

echo "==============================================================================="
echo "Figure 3"
echo "==============================================================================="
python3 figure3.py

echo "==============================================================================="
echo "Figure 4"
echo "==============================================================================="
python3 figure4.py

echo "==============================================================================="
echo "Figure 6"
echo "==============================================================================="
python3 figure6.py

echo "==============================================================================="
echo "Figure 7"
echo "==============================================================================="
python3 figure7.py

exit 0