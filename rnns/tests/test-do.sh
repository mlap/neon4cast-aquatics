#!/bin/bash

python rnn_train.py --epochs 1 --file-name test --variable do
python rnn_eval.py --png-name test --model-name test --start -100 --predict-window 8
python rnn_forecast.py --model-name test --predict-window 8 --start-date 2021-05-01 --end-date 2021-05-08
