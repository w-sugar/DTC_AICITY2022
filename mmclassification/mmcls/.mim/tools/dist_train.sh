#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29501}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} --resume-from '/home/qsh/mmclassification/tools/work_dirs/alladd2_resnext_auto/epoch_17.pth'
