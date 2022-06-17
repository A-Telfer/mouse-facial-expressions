#!/bin/bash

export EPOCHS=50
export SHUFFLES=5
export TRAIN_RATIO=0.95

python benchmark/train.py --save-dir benchmark-augmentation/none --augmentation none
python benchmark/train.py --save-dir benchmark-augmentation/baseline --augmentation baseline
python benchmark/train.py --save-dir benchmark-augmentation/randaug --augmentation randaug
python benchmark/train.py --save-dir benchmark-augmentation/trivial-wide --augmentation trivial-wide
