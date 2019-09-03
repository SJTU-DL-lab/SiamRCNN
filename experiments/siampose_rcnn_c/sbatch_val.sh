#!/bin/bash
#SBATCH -J siam_rcnn_val
#SBATCH -p gpu
#SBATCH --output=siamrcnn_val.log
#SBATCH --error=siamrcnn_val.err
#SBATCH --gres=gpu:1
date
module load anaconda2/5.3.0 cuda/9.0 cudnn/7.3.0
source activate pytorch0.4

# ROOT=/cluster/home/it_stu2/SiamPose
# ROOT=`git rev-parse --show-toplevel`
# export PYTHONPATH=$ROOT:$PYTHONPATH
ROOT=../../
export PYTHONPATH=$ROOT:$PYTHONPATH
# export PYTHONPATH=$PWD:$PYTHONPATH

mkdir -p logs

python -u $ROOT/tools/val_siamrcnn.py \
    --config=config.json -b 1 \
    -j 4 --debug \
    --epochs 200 --pretrained ../siampose_rcnn_base/snapshot_0902/checkpoint_e19.pth \
    --log logs/log_test.txt \
    --log-dir test_logs \
    2>&1 | tee logs/train_test.log

