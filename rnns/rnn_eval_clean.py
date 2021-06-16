import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from utils import *
from copy import deepcopy
import argparse

# Argument parsing block; to get help on this from CL run `python tune_sb3.py -h`
parser = argparse.ArgumentParser()
parser.add_argument("--csv-name", type=str, default="POSE_data.csv", help="Name of CSV to use")
parser.add_argument(
    "--png-name", type=str, default="trash", help="Name to save png"
)
parser.add_argument(
    "--model-name",
    type=str,
    default="64_lstm_64_hidden_BARC_final_DO",
    help="Name of model to load",
)
parser.add_argument("--train-window", type=int, default=21)
parser.add_argument("--variable", type=str, default="wt", help="Name of variable being predicted (wt/do)")
parser.add_argument("--predict-window", type=int, default=7)
parser.add_argument("--start", type=int, default=-29)
parser.add_argument("--n-layers", type=int, default=2)
args = parser.parse_args()

def main():
    df = get_data(args.csv_name)
    variables = get_variables(args)
    data = df[variables]
    # Normalizing data to -1, 1 scale; this improves performance of neural nets
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_scaled = scaler.fit_transform(data)
    # Creating a sequence to condition the LSTM cells
    condition_seq = create_sequence(
        data_scaled[: args.start], args.train_window
    )
    # Indexing the appropriate data
    end = args.start + args.train_window + args.predict_window
    evaluation_data = data[args.start : end]
    # Evaluating the data
    means, stds = evaluate(evaluation_data, condition_seq, args, scaler)
    # Plotting the data
    data_len = len(evaluation_data)
    start_idx = data_len - args.predict_window
    end_idx = data_len
    plot(evaluation_data, means, stds, args, start_idx, end_idx)


if __name__ == "__main__":
    main()