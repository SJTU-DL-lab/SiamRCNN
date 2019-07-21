ROOT=`git rev-parse --show-toplevel`
export PYTHONPATH=$ROOT:$PYTHONPATH
export PYTHONPATH=$PWD:$PYTHONPATH

mkdir -p logs

python -u $ROOT/tools/train_siamrcnn.py \
    --config=config.json -b 4 \
    -j 8 --pretrain /home/yaosy/Diskb/projects/SiamPose/experiments/siampose_osct/snapshot/checkpoint_e199.pth \
    --epochs 200 \
    --log logs/log.txt \
    2>&1 | tee logs/train.log

#bash test_all.sh -s 1 -e 20 -d VOT2018 -g 4
#bash test_all.sh -s 1 -e 20 -d VOT2018 -g 1
