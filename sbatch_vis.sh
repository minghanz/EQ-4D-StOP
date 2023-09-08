#!/bin/bash
#SBATCH --job-name 4DSVI
#SBATCH --nodes=1
#SBATCH --time=0:30:00
#SBATCH --account=hpeng1
#######SBATCH --mail-type=END,FAIL

#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3g

source ~/anaconda3/etc/profile.d/conda.sh
conda activate 4dpls

### This is to print the host name
ipnip=$(hostname -i)
echo ipnip=$ipnip


### Run inference (you do not need to complete the full dataset)
python -u test_models.py --log results/Log_2023-02-08_23-31-15 --idx 2 --name vis_vote_1100 --vis

### Generate the visualization (specify sequence_number, frame_number, and log_dir in final_vis_4d.py)
python -u vis_scripts/final_vis_4d.py

### Inspect the visualization
cd test/Log_2023-02-08_23-31-15/vis_vote_1100/visualizations/vis_4d_08_0000028_pan_4D-StOP
python -m http.server 6008