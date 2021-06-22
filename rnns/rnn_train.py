import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
import optuna
import pandas as pd
import numpy as np
from random import shuffle
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy
from utils import *
import argparse

# Argument parsing block; to get help on this from CL run `python tune_sb3.py -h`
parser = argparse.ArgumentParser()
parser.add_argument("--csv-name", type=str, default="POSE_data", help="Name of CSV to use")
parser.add_argument(
    "--file-name",
    type=str,
    default="trash_model_dist",
    help="Name to save model",
)
parser.add_argument("--epochs", type=int, default=25, help="Number of Epochs")
parser.add_argument("--variable", type=str, default="do", help="Name of variable being predicted (wt/do)")
args = parser.parse_args()

# Edit hyperparameters here
params = {
    "learning_rate": 0.000001,
    "train_window": 21,
    "hidden_dim": 64,
    "n_layers": 2,
}

def main(device):
    df = get_data(args.csv_name)
    params_etcs = {"variable": args.variable, "csv_name": args.csv_name}
    variables = get_variables(params_etcs)
    training_data = df[variables]
    # Normalizing data to -1, 1 scale; this improves performance of neural nets
    scaler = MinMaxScaler(feature_range=(-1, 1))
    training_data_normalized = scaler.fit_transform(training_data)
    # Training the model
    train(training_data_normalized, params, args, device, save_flag=True)


if __name__ == "__main__":
    torch.cuda.set_device(0)
    print("Active Cuda Device: GPU ", torch.cuda.current_device())
    main(torch.cuda.current_device())
