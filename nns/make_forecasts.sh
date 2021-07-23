#!/bin/bash

eval_forecast(){
  cd $1
  python rnn_train.py --epochs 200 --model-name trash --variable do 
  python rnn_eval.py --png-name test_do --model-name trash
  mv trash.png ../trash_$1.png
  cd ..
}

eval_forecast lstm_dropout
eval_forecast lstm_dist
eval_forecast mlp_dropout
eval_forecast mlp_dist
eval_forecast lstm_autoencoder
