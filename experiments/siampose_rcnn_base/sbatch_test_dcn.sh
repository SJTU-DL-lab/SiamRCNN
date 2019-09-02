#!/bin/bash
#SBATCH -J dcn_test
#SBATCH -p gpu
#SBATCH --output=dcn_test.out
#SBATCH --error=dcn_test.err
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

python -u $ROOT/experiments/siampose_rcnn_c/dcn_test.py

