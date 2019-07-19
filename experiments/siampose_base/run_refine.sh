ROOT=`git rev-parse --show-toplevel`
export PYTHONPATH=$ROOT:$PYTHONPATH

mkdir -p logs

python -u $ROOT/tools/train_siampose_refine.py \
    --config=config_refine.json -b 4 \
    -j 8 --resume checkpoint_e6.pth \
    --epochs 20 \
    --log logs/log.txt \
    2>&1 | tee logs/train.log

#bash test_all.sh -s 1 -e 20 -d VOT2018 -g 4
#bash test_all.sh -s 1 -e 20 -d VOT2018 -g 1
