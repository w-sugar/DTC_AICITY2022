#!/usr/bin/env bash

GPUS=$1
PORT=${PORT:-29501}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py ./mmclassification/configs/efficientnet/efficientnet-b0_8xb32-01norm_in1k.py --launcher pytorch ${@:3} --no-validate --work-dir ./mmclassification/work_dirs/b0

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py ./mmclassification/configs/efficientnet/efficientnet-b2_8xb32-01norm_in1k.py --launcher pytorch ${@:3} --no-validate --work-dir ./mmclassification/work_dirs/b2

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py ./mmclassification/configs/resnest/resnest50_32xb64_in1k.py --launcher pytorch ${@:3} --no-validate --work-dir ./mmclassification/work_dirs/resnest50

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py ./mmclassification/configs/resnest/resnest101_32xb64_in1k.py --launcher pytorch ${@:3} --no-validate --work-dir ./mmclassification/work_dirs/resnest101