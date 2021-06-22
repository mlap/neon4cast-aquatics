#!/bin/bash
source tests/check_del_file.sh

python rnn_tune.py --n-trials 5 --variable wt
check_del_file studies/trash.db

python rnn_tune.py --n-trials 5
check_del_file studies/trash.db