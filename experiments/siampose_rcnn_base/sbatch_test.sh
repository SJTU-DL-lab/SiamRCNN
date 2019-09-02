#!/bin/bash
#SBATCH -J siam_rcnn_test
#SBATCH -p gpu
#SBATCH --output=siampose_test.log
#SBATCH --error=siampose_test.err
#SBATCH --gres=gpu:1
date
module load anaconda2/5.3.0 cuda/9.0 cudnn/7.3.0
source activate deepimit

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
    --epochs 200 --pretrained ./snapshot_0819/checkpoint_e199.pth \
    --log logs/log_test.txt \
    --log-dir test_logs \
    2>&1 | tee logs/train_test.log

