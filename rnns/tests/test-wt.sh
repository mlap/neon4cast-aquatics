#!/bin/bash
source tests/check_del_file.sh

train_eval_forecast(){
  python rnn_train.py --epochs 1 --file-name test_wt --variable wt --csv-name $1
  python rnn_eval.py --png-name test_wt --model-name test_wt --start -100 --predict-window 8
  python rnn_forecast.py --model-name test_wt --predict-window 8 --start-date 2021-05-01 --end-date 2021-05-08 --png-name test_wt_forecast
  
  check_del_file models/test_wt.pkl
  check_del_file models/test_wt.yaml
  check_del_file test_wt.png
  check_del_file test_wt_forecast.png
}

train_eval_forecast POSE_waterT_data
train_eval_forecast BARC_waterT_data
