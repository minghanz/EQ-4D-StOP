#!/bin/bash
#SBATCH --job-name 4DSTE
#SBATCH --nodes=1
#SBATCH --time=200:00:00
#SBATCH --account=hpeng1
#SBATCH --mail-type=END,FAIL

#SBATCH --partition=spgpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=2

#SBATCH --mem-per-cpu=6g

source ~/anaconda3/etc/profile.d/conda.sh
conda activate 4dpls

### This is to print the host name
ipnip=$(hostname -i)
echo ipnip=$ipnip

### Run inference
python -u test_models.py --log results/Log_yyyy-mm-dd_hh-mm-ss --name NAME

### Associate across tracklets
python stitch_tracklets.py --predictions test/Log_yyyy-mm-dd_hh-mm-ss/NAME --nuscene --dataset nuScenes_like_SKitti --data_cfg nuScenes_like_SKitti/semantic-kitti.yaml

### Evaluation
python utils/evaluate_4dpanoptic_nuscene.py \
--predictions=test/Log_yyyy-mm-dd_hh-mm-ss/NAME/stitch4/ \
--output=test/Log_yyyy-mm-dd_hh-mm-ss/NAME/output_metrics.log