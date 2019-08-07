#!/bin/bash
#SBATCH -J siam_rcnn
#SBATCH -p gpu
#SBATCH --output=siampose_ct
#SBATCH --error=siampose_cterr
#SBATCH --nodelist=node5
#SBATCH --gres=gpu:4
date
module load anaconda2/5.3.0 cuda/9.0 cudnn/7.3.0
source activate pytorch0.4

# ROOT=/cluster/home/it_stu2/SiamPose
# ROOT=`git rev-parse --show-toplevel`
# export PYTHONPATH=$ROOT:$PYTHONPATH
ROOT=/cluster/home/it_stu1/bdclub/SiamRCNN/
export PYTHONPATH=$ROOT:$PYTHONPATH
# export PYTHONPATH=$PWD:$PYTHONPATH

mkdir -p logs

python -u $ROOT/tools/trainval_siamrcnn.py \
    --config=config.json -b 48 \
    -j 8 --save_freq 10 \
    --pretrained snapshot_full_img/checkpoint_e88.pth \
    --epochs 200 --hm_hp_weight 100 \
    --mse_loss \
    --log-dir board/show \
    --log logs/log.txt \
    2>&1 | tee logs/train.log

