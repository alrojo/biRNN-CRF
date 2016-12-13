#!/bin/sh
python train.py plain 0
python train.py plain 1
python train.py plain_bn 0
python train.py plain_bn 1
python train.py plain_dropout 0
python train.py plain_dropout 1
python train.py plain_bn_dropout 0
python train.py plain_bn_dropout 1
