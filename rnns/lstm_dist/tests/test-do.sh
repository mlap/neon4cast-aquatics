#!/bin/bash
source tests/check_del_file.sh

train_eval_forecast(){
  python rnn_train.py --epochs 1 --model-name test_do --variable do --csv-name $1 --network $2
  python rnn_eval.py --png-name test_do --model-name test_do --start -100 --predict-window 8
  python rnn_forecast.py --model-name test_do --predict-window 8 --start-date 2021-05-01 --end-date 2021-05-08 --png-name test_do_forecast

  check_del_file models/test_do.pkl
  check_del_file models/test_do.yaml
  check_del_file test_do.png
  check_del_file test_do_forecast.png
}

train_eval_forecast BARC_data lstm
train_eval_forecast BARC_data gru
train_eval_forecast POSE_data lstm
train_eval_forecast POSE_data gru