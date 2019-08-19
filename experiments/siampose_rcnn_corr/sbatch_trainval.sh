#!/bin/bash
#SBATCH -J siam_rcnn_corr
#SBATCH -p gpu
#SBATCH --output=siampose_corr.out
#SBATCH --error=siampose_corr.err
#SBATCH --gres=gpu:4
#SBATCH --nodelist=node11
date
module load anaconda2/5.3.0 cuda/9.0 cudnn/7.3.0
source activate pytorch0.4

# ROOT=/cluster/home/it_stu2/SiamPose
# ROOT=`git rev-parse --show-toplevel`
ROOT=../../
export PYTHONPATH=$ROOT:$PYTHONPATH
# ROOT=/cluster/home/u1/sjtu_bdclub/ysy_github/SiamRCNN/
# export PYTHONPATH=$ROOT:$PYTHONPATH
# export PYTHONPATH=$PWD:$PYTHONPATH

mkdir -p logs

python -u $ROOT/tools/trainval_siamrcnn_corr.py \
    --config=config.json -b 64 \
    -j 4 --save_freq 1 \
    --epochs 100 --hm_hp_weight 1.5 \
    --log-dir board/show \
    --log logs/log.txt \
    --pretrained ./snapshot/checkpoint_e99.pth
    # --pretrained snapshot_full_img/checkpoint_e88.pth
    2>&1 | tee logs/train.log

