#!/bin/bash
#SBATCH -J siam_rcnn_test
#SBATCH -p gpu
#SBATCH --output=siamrcnn_test.out
#SBATCH --error=siamrcnn_test.err
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node5
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

python -u $ROOT/tools/trainval_siamrcnn.py \
    --config=config.json -b 1 \
    -j 4 --save_freq 2 \
    --epochs 200 --hm_hp_weight 1.5 \
    --log-dir board/show_test \
    --save_dir snapshot_test \
    --log logs/log_test.txt
    --pretrained ../siampose_rcnn_base/snapshot_0902/checkpoint_e19.pth
    # --pretrained snapshot_0815/checkpoint_e199.pth
    # --pretrained snapshot_full_img/checkpoint_e88.pth
    2>&1 | tee logs/train.log

date
