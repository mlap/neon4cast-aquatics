# This needs to be reworked
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
parser.add_argument("--variable", type=str, default="do", help="Name of variable being predicted (wt/do)")
parser.add_argument("--csv-name", type=str, default="POSE_data", help="Name of CSV to use")
parser.add_argument(
    "--study-name", type=str, default="trash", help="Study name"
)
parser.add_argument(
    "--n-trials", type=int, default=int(25), help="Number of tuning trials"
)
parser.add_argument("--epochs", type=int, default=1, help="Number of Epochs")
parser.add_argument(
    "--predict-window",
    type=int,
    default=7,
    help="How many days in advance to predict",
)
args = parser.parse_args()


def get_params(trial):
    params = {
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1),
        "train_window": trial.suggest_categorical("train_window", [21, 28, 35, 42, 49]),
        "n_layers": trial.suggest_categorical(
            "n_layers", [2, 3, 4]
        ),
        "hidden_dim": trial.suggest_categorical(
            "hidden_dim", [8, 16, 32]
        ),
    }
    return params


def score_model(model, params):
    #model.cpu()
    df = get_data(args.csv_name)
    params_etcs = {"variable": args.variable, "csv_name": args.csv_name}
    #params.update(params_etcs)
    variables = get_variables(params)
    data = df[variables]
    
    # Normalizing data to -1, 1 scale; this improves performance of neural nets
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_normalized = scaler.fit_transform(data)
    
    # Conditioning lstm cells
    condition_seq = create_sequence(
        data_normalized[: -params["train_window"] - args.predict_window], params["train_window"]
    )
    evaluation_data = data_normalized[
        -params["train_window"] - args.predict_window : -args.predict_window
    ]
    means, stds = evaluate(evaluation_data, condition_seq, args, scaler, params, model)

    DO_targets = (
        data[["dissolvedOxygen"]][-args.predict_window:].to_numpy().reshape(-1)
    )
    WT_targets = (
        data[["groundwaterTempMean"]][-args.predict_window:]
        .to_numpy()
        .reshape(-1)
    )
    objective = (
        ((means[:, 2] - DO_targets) ** 2).mean()
        + ((means[:, 2] + stds[:, 2] - DO_targets) ** 2).mean()
        + ((means[:, 0] - WT_targets) ** 2).mean()
        + ((means[:, 0] + stds[:, 0] - WT_targets) ** 2).mean()
    )

    return objective


def train_model(params, device):
    df = get_data(args.csv_name)
    params_etcs = {"variable": args.variable, "csv_name": args.csv_name}
    variables = get_variables(params_etcs)
    training_data = df[variables]
    # Normalizing data to -1, 1 scale; this improves performance of neural nets
    scaler = MinMaxScaler(feature_range=(-1, 1))
    training_data_normalized = scaler.fit_transform(training_data)
    # Training the model
    params.update(params_etcs)
    model = train(training_data_normalized, params, args, device, save_flag=False)
    return model


def objective(trial):
    params = get_params(trial)
    model = train_model(params, device=0)
    objective = score_model(model, params)
    del model
    return objective


if __name__ == "__main__":
    torch.cuda.set_device(0)
    print("GPU Device: ", torch.cuda.current_device())
    storage_name = f"sqlite:///studies/{args.study_name}.db"
    # Sampling from hyperparameters using TPE over 50 trials
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(
        study_name=args.study_name,
        sampler=sampler,
        direction="minimize",
        storage=storage_name,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=25, catch=(ValueError, RuntimeError))
    # Reporting best trial and making a quick plot to examine hyperparameters
    trial = study.best_trial
    print(f"Best hyperparams: {trial.params}")
