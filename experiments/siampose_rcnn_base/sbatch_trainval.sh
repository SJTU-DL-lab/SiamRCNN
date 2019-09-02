#!/bin/bash
#SBATCH -J siam_rcnn_base_0902
#SBATCH -p gpu
#SBATCH --output=siamrcnn_base_0902.out
#SBATCH --error=siamrcnn_base_0902.err
#SBATCH --gres=gpu:4
#SBATCH --nodelist=node2
date
module load anaconda2/5.3.0 cuda/9.0 cudnn/7.3.0
source activate torch0.4

# ROOT=/cluster/home/it_stu2/SiamPose
# ROOT=`git rev-parse --show-toplevel`
# export PYTHONPATH=$ROOT:$PYTHONPATH
ROOT=../../
export PYTHONPATH=$ROOT:$PYTHONPATH
# export PYTHONPATH=$PWD:$PYTHONPATH

mkdir -p logs

python -u $ROOT/tools/trainval_siamrcnn_base.py \
    --config=config.json -b 64 \
    -j 4 --save_freq 2 \
    --epochs 20 --hm_hp_weight 1.5 \
    --log-dir board/show_0902 \
    --save_dir snapshot_0902 \
    --log logs/log_0902.txt \
    # --resume snapshot_0819/checkpoint_e63.pth
    # --pretrained snapshot_0815/checkpoint_e199.pth
    # --pretrained snapshot_full_img/checkpoint_e88.pth
    2>&1 | tee logs/train.log

date
