#! /usr/bin/bash 
cd /data2/dl/grp03/New\ folder/
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate grp10_venv1
set -x
python3 train.py 

