##!/usr/bin/env bash

DTU_TESTING="/data1/local_userdata/wangshaoqian/DTU/dtu_testing/"

TANK_TESTING='/data1/local_userdata/wangshaoqian/tankandtemples/'

CKPT_FILE="./checkpoints/TANK_train_on_dtu.ckpt"

OUT_DIR='Effi-MVS_result'

if [ ! -d $OUT_DIR ]; then
    mkdir -p $OUT_DIR
fi
##DTU
python test.py --dataset=general_eval --batch_size=1 --testpath=$DTU_TESTING  --ndepths=48 --CostNum=4 --numdepth=384 --testlist lists/dtu/test.txt --loadckpt $CKPT_FILE --outdir $OUT_DIR --data_type dtu \
              --num_view=5 --interval_scale=0.265
##tank
#python test.py --dataset=tank --batch_size=1 --testpath=$TANK_TESTING  --ndepths=96 --CostNum=4 --numdepth=384 --loadckpt $CKPT_FILE --outdir $OUT_DIR --data_type tank \
#              --num_view=7 --interval_scale=0.265
