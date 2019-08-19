#!/bin/bash
#SBATCH -J siam_rcnn_featEn_0819
#SBATCH -p gpu
#SBATCH --output=siamrcnn_bk_0819.out
#SBATCH --error=siamrcnn_bk_0819.err
#SBATCH --gres=gpu:4
#SBATCH --nodelist=node11
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
    --config=config.json -b 64 \
    -j 4 --save_freq 2 \
    --epochs 100 --hm_hp_weight 1.5 \
    --log-dir board/show_0819 \
    --save_dir snapshot_0819 \
    --log logs/log_0819.txt \
    --pretrained snapshot/checkpoint_e99.pth
    # --pretrained snapshot_0815/checkpoint_e199.pth
    # --pretrained snapshot_full_img/checkpoint_e88.pth
    2>&1 | tee logs/train.log

date
