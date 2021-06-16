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
parser.add_argument(
    "--png-name", type=str, default="trash_forecast", help="Name to save png"
)
parser.add_argument(
    "--model-name",
    type=str,
    default="trash_model_dist",
    help="Name of model to load",
)
parser.add_argument("--start-date", type=str, default="2021-05-01")
parser.add_argument("--end-date", type=str, default="2021-05-07")
parser.add_argument("--predict-window", type=int, default=7, help="How long of a forecast to make")
args = parser.parse_args()


def main():
    params_etcs = load_etcs(args.model_name)
    df = get_data(params_etcs["csv_name"])
    variables = get_variables(params_etcs)
    data = df[variables]
    # Normalizing data to -1, 1 scale; this improves performance of neural nets
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_scaled = scaler.fit_transform(data)
    condition_seq = create_sequence(
        data_scaled[: -params_etcs["train_window"]], params_etcs["train_window"]
    )
    # Indexing the appropriate data
    evaluation_data = data[-params_etcs["train_window"]:]
    # Evaluating the data
    means, stds = evaluate(evaluation_data, condition_seq, args, scaler, params_etcs)
    
    make_forecast(args, params_etcs, means, stds)
    
    data_len = len(evaluation_data)
    start_idx = data_len + 1
    end_idx = data_len + 1 + args.predict_window
    plot(evaluation_data, means, stds, args, params_etcs, start_idx, end_idx)

if __name__ == "__main__":
    main()
