ROOT=`git rev-parse --show-toplevel`
export PYTHONPATH=$ROOT:$PYTHONPATH

mkdir -p logs

python -u $ROOT/tools/train_siampose_directReg.py \
    --config=config_refine.json -b 16 \
    -j 8 --method rel1 \
    --epochs 20 --resume ./snapshot/checkpoint_e4.pth \
    --log logs/log.txt \
    2>&1 | tee logs/train.log

#bash test_all.sh -s 1 -e 20 -d VOT2018 -g 4
#bash test_all.sh -s 1 -e 20 -d VOT2018 -g 1
