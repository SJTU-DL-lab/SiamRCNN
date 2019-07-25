#!/bin/bash
#SBATCH -J siam_rcnn
#SBATCH -p gpu
#SBATCH --output=siampose_ct
#SBATCH --error=siampose_cterr
#SBATCH --nodelist=node6
#SBATCH --gres=gpu:1
date
module load anaconda3/5.3.0 cuda/9.0 cudnn/7.3.0
source activate pytorch0.4

# ROOT=/cluster/home/it_stu2/SiamPose
# ROOT=`git rev-parse --show-toplevel`
# export PYTHONPATH=$ROOT:$PYTHONPATH
ROOT=/cluster/home/it_stu1/bdclub/SiamRCNN/
export PYTHONPATH=$ROOT:$PYTHONPATH
# export PYTHONPATH=$PWD:$PYTHONPATH

mkdir -p logs

python -u $ROOT/tools/train_siamrcnn.py \
    --config=config.json -b 1 \
    -j 8 --resume ./snapshot/checkpoint_e22.pth \
    --epochs 200 \
    --log logs/log.txt \
    2>&1 | tee logs/train.log

