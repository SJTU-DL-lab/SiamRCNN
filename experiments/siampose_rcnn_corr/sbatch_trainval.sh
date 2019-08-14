#!/bin/bash
#SBATCH -J siam_rcnn_corr_0814
#SBATCH -p gpu
#SBATCH --output=siamrcnn_corr_0814.out
#SBATCH --error=siamrcnn_corr_0814.err
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node11
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

python -u $ROOT/tools/trainval_siamrcnn_corr.py \
    --config=config.json -b 16 \
    -j 4 --save_freq 5 \
    --epochs 200 --hm_hp_weight 1.5 \
    --save_dir snapshot_0814 \
    --log-dir board/show_0814 \
    --log logs/log_0814.txt \
    --pretrained ./snapshot_0811/checkpoint_e99.pth
    # --pretrained snapshot_full_img/checkpoint_e88.pth
    2>&1 | tee logs/train.log

