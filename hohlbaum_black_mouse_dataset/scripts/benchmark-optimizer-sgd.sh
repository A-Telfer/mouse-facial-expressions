#!/bin/bash

export EPOCHS=50
export SHUFFLES=5
export TRAIN_RATIO=0.95
export OPTIMIZER=sgd

python benchmark/train.py --save-dir benchmark-optimizer/sgd-lr05 --learning-rate 0.05
python benchmark/train.py --save-dir benchmark-optimizer/sgd-lr01 --learning-rate 0.01
python benchmark/train.py --save-dir benchmark-optimizer/sgd-lr001 --learning-rate 0.001
python benchmark/train.py --save-dir benchmark-optimizer/sgd-lr0001 --learning-rate 0.0001
python benchmark/train.py --save-dir benchmark-optimizer/sgd-lr00001 --learning-rate 0.00001


