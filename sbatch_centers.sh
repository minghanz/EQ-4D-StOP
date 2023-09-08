#!/bin/bash
#SBATCH --job-name 4DSCT
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --account=hpeng1
#SBATCH --mail-type=END,FAIL

#SBATCH --partition=standard
#SBATCH --cpus-per-task=1

#SBATCH --mem-per-cpu=10g

# conda init bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate 4dpls

### This is to print the host name
ipnip=$(hostname -i)
echo ipnip=$ipnip

### Generate center labels for SemanticKITTI
python utils/create_center_label.py

### Generate center labels for converted nuScenes in SemanticKITTI format
python utils/create_center_label.py --nuscene