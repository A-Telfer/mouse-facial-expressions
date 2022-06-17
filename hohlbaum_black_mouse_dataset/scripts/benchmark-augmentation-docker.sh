#!/bin/bash
python benchmark/train.py --save-model --save-dir benchmark-augmentation/none --epochs 50 --shuffles 5 --train-ratio 0.95 --augmentation none
python benchmark/train.py --save-model --save-dir benchmark-augmentation/baseline --epochs 50 --shuffles 5 --train-ratio 0.95 --augmentation baseline
python benchmark/train.py --save-model --save-dir benchmark-augmentation/randaug --epochs 50 --shuffles 5 --train-ratio 0.95 --augmentation randaug
python benchmark/train.py --save-model --save-dir benchmark-augmentation/trivial-wide --epochs 50 --shuffles 5 --train-ratio 0.95 --augmentation trivial-wide
