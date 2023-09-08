#!/bin/bash
#SBATCH --job-name 4DSTR
#SBATCH --nodes=1
#SBATCH --time=200:00:00
#SBATCH --account=hpeng1
#SBATCH --mail-type=END,FAIL

#SBATCH --partition=spgpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=6

#SBATCH --mem-per-cpu=4g

# conda init bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate 4dpls

### This is to print the host name
ipnip=$(hostname -i)
echo ipnip=$ipnip

### EQ-4D-StOP
## First stage
python -u train_SemanticKitti.py --eq --fdim 64 --kanchor 4
## Second stage
# python -u train_SemanticKitti.py --eq --fdim 64 --kanchor 4 -l Log_yyyy-mm-dd_hh-mm-ss

### 4D-StOP baseline of the same feature map size
## First stage
python -u train_SemanticKitti.py --fdim 256
## Second stage
# python -u train_SemanticKitti.py --fdim 256 -l Log_yyyy-mm-dd_hh-mm-ss

### Specify `--train_val` for official leaderboard model, `--nuscene` for nuscenes.