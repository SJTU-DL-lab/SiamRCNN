ROOT=`git rev-parse --show-toplevel`
export PYTHONPATH=$ROOT:$PYTHONPATH

mkdir -p logs

python -u $ROOT/tools/train_siampose_bibranch.py \
    --config=config_bibranch.json -b 16 \
    -j 8 --resume ./siampose_bibrach_0630/checkpoint_e92.pth \
    --epochs 20 \
    --log logs/log.txt \
    2>&1 | tee logs/train.log

#bash test_all.sh -s 1 -e 20 -d VOT2018 -g 4
#bash test_all.sh -s 1 -e 20 -d VOT2018 -g 1
