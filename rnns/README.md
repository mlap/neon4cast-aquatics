Run `pip install -r requirements.txt` to get all the right python packages. Recommended you do this in a virtual environment.

`rnn_train.py` trains a LSTM model on the data in csv format. 

`rnn_eval.py` evaluates a trained LSTM model by creating a matplotlib plot that compares actual data vs. predictions. 

`rnn_forecast.py` creates a forecast csv and a matplotlib plot that visualizes the forecast.

`rnn_tune.py` runs a optuna-based tuning round to find a working set of hyperparameters.

In all cases here, the user can specify options through command line flags. Run `python rnn_*.py -h` to get more information on the available command line flags. 
