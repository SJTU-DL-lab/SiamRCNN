#!/bin/bash
#SBATCH -J siam_rcnn_dcn
#SBATCH -p gpu
#SBATCH --output=siampose_ct
#SBATCH --error=siampose_cterr
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node1
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

python -u $ROOT/tools/trainval_siamrcnn.py \
    --config=config.json -b 1 \
    -j 1 --save_freq 1 \
    --epochs 200 --hm_hp_weight 1.5 \
    --log-dir board/show \
    --log logs/log.txt \
    # --pretrained snapshot/checkpoint_e99.pth
    # --pretrained snapshot_full_img/checkpoint_e88.pth
    2>&1 | tee logs/train.log

