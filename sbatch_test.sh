#!/bin/bash
#SBATCH --job-name 4DSTE
#SBATCH --nodes=1
#SBATCH --time=100:00:00
#SBATCH --account=hpeng1
#SBATCH --mail-type=END,FAIL

#SBATCH --partition=spgpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=2

#SBATCH --mem-per-cpu=6g

# conda init bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate 4dpls

### This is to print the host name
ipnip=$(hostname -i)
echo ipnip=$ipnip

### Run inference 
### (specify `--test` for test set, `--frame 1` for single frame mode, `--idx INTEGER` for checkpoints other than current_chkp.tar)
### (--name NAME is optional. Not specifying it results in an output folder named by the timestamp)
python -u test_models.py --log results/Log_yyyy-mm-dd_hh-mm-ss --name NAME

### Associate across tracklets (specify `--split test` for test set, `--n_test_frames 1` for single frame mode)
python stitch_tracklets.py --predictions test/Log_yyyy-mm-dd_hh-mm-ss/NAME 

### Evaluation on validation set
python utils/evaluate_4dpanoptic.py \
--predictions=test/Log_yyyy-mm-dd_hh-mm-ss/NAME/stitch4/ \
--output=test/Log_yyyy-mm-dd_hh-mm-ss/NAME/output_metrics.log

### Evaluation on validation set for single frame mode
python utils/evaluate_panoptic.py \
--predictions=test/Log_yyyy-mm-dd_hh-mm-ss/NAME/stitch1/ \
--output=test/Log_yyyy-mm-dd_hh-mm-ss/NAME/output_metrics_3D

### Test set result submission
cd test/Log_yyyy-mm-dd_hh-mm-ss/NAME/stitch4
zip -r submission.zip .