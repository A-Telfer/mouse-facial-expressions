#!/bin/bash
python benchmark/train.py --save-model --save-dir benchmark --epochs 50 --shuffles 5 --train-ratio 0.95
