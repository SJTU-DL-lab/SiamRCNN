#!/bin/bash
#SBATCH -J siam_rcnn_test
#SBATCH -p gpu
#SBATCH --output=siampose_test.log
#SBATCH --error=siampose_test.err
#SBATCH --nodelist=node11
#SBATCH --gres=gpu:1
date
module load anaconda2/5.3.0 cuda/9.0 cudnn/7.3.0
source activate deepimit

# ROOT=/cluster/home/it_stu2/SiamPose
# ROOT=`git rev-parse --show-toplevel`
# export PYTHONPATH=$ROOT:$PYTHONPATH
ROOT=/cluster/home/u1/sjtu_bdclub/ysy_github/SiamRCNN/
export PYTHONPATH=$ROOT:$PYTHONPATH
# export PYTHONPATH=$PWD:$PYTHONPATH

mkdir -p logs

python -u $ROOT/tools/val_siamrcnn.py \
    --config=config.json -b 1 \
    -j 4 --debug \
    --epochs 200 --pretrained snapshot_0811/checkpoint_e99.pth \
    --log logs/test_log.txt \
    --log-dir test_logs \
    2>&1 | tee logs/test.log

