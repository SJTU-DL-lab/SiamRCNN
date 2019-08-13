#!/bin/bash
#SBATCH -J siam_rcnn_corr_md
#SBATCH -p gpu
#SBATCH --output=siamrcnn_corr_md.out
#SBATCH --error=siamrcnn_corr_md.err
#SBATCH --gres=gpu:2
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
    --config=config.json -b 32 \
    -j 4 --save_freq 5 \
    --epochs 200 --hm_hp_weight 1.5 \
    --save_dir snaphot_0812_modi \
    --log-dir board/show_0812_modi \
    --log logs/log_0812_modi.txt \
    --resume ./snaphot_0812_modi/checkpoint_e4.pth
    # --pretrained snapshot_full_img/checkpoint_e88.pth
    2>&1 | tee logs/train.log

