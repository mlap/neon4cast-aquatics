import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
import optuna
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy
from rnn_utils import *
import argparse

# Argument parsing block; to get help on this from CL run `python tune_sb3.py -h`
parser = argparse.ArgumentParser()
parser.add_argument("--study-name", type=str,
                    default="trash", help="Study name")
parser.add_argument("--n-trials", type=int, default=int(25),
                    help="Number of tuning trials")
parser.add_argument("--epochs", type=int, default=1, help="Number of Epochs")
parser.add_argument("--prediction-window", type=int, default=7,
                    help="How many days in advance to predict")
args = parser.parse_args()


def get_params(trial):
    params = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1),
        'train_window': trial.suggest_categorical('train_window', [7, 14, 21]),
        'lstm_width': trial.suggest_categorical('lstm_width', [64, 128, 256, 512]),
        'hidden_width': trial.suggest_categorical('hidden_width', [64, 128, 256, 512]),
    }

    return params


def score_model(model, params):
    df = pd.read_csv("minlake_test.csv", delimiter=",", index_col=0)
    df = df[df.year > 2019].sort_values(['year', 'month', 'day'])
    training_data = df[["groundwaterTempMean", "uPARMean",
                        "dissolvedOxygen", "chlorophyll"]]
    # Normalizing data to -1, 1 scale; this improves performance of neural nets
    scaler = MinMaxScaler(feature_range=(-1, 1))
    training_data_normalized = scaler.fit_transform(training_data)
    fut_pred = args.prediction_window
    train_window = params['train_window']
    all_predictions = []
    test_inputs = training_data_normalized[-train_window -
                                           fut_pred:-fut_pred]
    for i in range(1):
        medians = np.array([])
        bottom = np.array([])
        top = np.array([])
        model.eval()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))
        for i in range(fut_pred):
            seq = torch.FloatTensor(test_inputs[-train_window:])
            with torch.no_grad():
                dist = build_dist(model, seq)
                samples = dist.rsample((1000,))
                scaled_samples = scaler.inverse_transform(samples)
                medians = np.append(medians, np.percentile(
                    scaled_samples, 50, axis=0)).reshape(-1, 4)
                bottom = np.append(bottom, np.percentile(
                    scaled_samples, 30, axis=0)).reshape(-1, 4)
                top = np.append(top, np.percentile(
                    scaled_samples, 70, axis=0)).reshape(-1, 4)

    DO_targets = training_data[["dissolvedOxygen"]
                               ][-fut_pred:].to_numpy().reshape(-1)
    WT_targets = training_data[["groundwaterTempMean"]
                               ][-fut_pred:].to_numpy().reshape(-1)
    import pdb; pdb.set_trace()
    objective = ((medians[:, 2] - DO_targets)**2).mean() + ((medians[:, 2] - bottom[:, 2] - DO_targets)**2).mean() + \
        ((medians[:, 2] + top[:, 2] - DO_targets)**2).mean() + \
        ((medians[:, 0] - WT_targets)**2).mean() + ((medians[:, 0] + top[:, 0] - WT_targets)**2).mean() + \
        ((medians[:, 0] - bottom[:, 0] - WT_targets)**2).mean()

    return objective


def train_model(params):
    df = pd.read_csv("minlake_test.csv", delimiter=",", index_col=0)
    df = df[df.year > 2019].sort_values(['year', 'month', 'day'])
    training_data = df[["groundwaterTempMean", "uPARMean",
                        "dissolvedOxygen", "chlorophyll"]]
    # Normalizing data to -1, 1 scale; this improves performance of neural nets
    scaler = MinMaxScaler(feature_range=(-1, 1))
    training_data_normalized = scaler.fit_transform(training_data)

    model = LSTM(
        input_size=4, hidden_layer_size=params['lstm_width'], fc_size=params['hidden_width'], output_size=14)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params['learning_rate'])

    epochs = args.epochs
    train_window = params['train_window']

    train_seq = create_sequence(training_data_normalized, train_window)

    for i in range(epochs):
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))
        for seq, targets in train_seq:
            optimizer.zero_grad()
            model.float()
            seq = torch.from_numpy(seq)
            dist = build_dist(model, seq)

            targets = torch.from_numpy(targets).view(len(targets), -1).float()
            single_loss = -dist.log_prob(targets)
            model.hidden_cell = (
                model.hidden_cell[0].detach(), model.hidden_cell[1].detach())
            single_loss.backward()
            optimizer.step()

        if i % 25 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

    return model


def objective(trial):
    params = get_params(trial)
    model = train_model(params)
    objective = score_model(model, params)

    return objective


if __name__ == "__main__":
    torch.cuda.set_device(0)
    print("GPU Device: ", torch.cuda.current_device())
    storage_name = f"sqlite:///studies/{args.study_name}.db"
    # Sampling from hyperparameters using TPE over 50 trials
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(study_name=args.study_name, sampler=sampler,
                                direction='minimize', storage=storage_name,
                                load_if_exists=True)
    study.optimize(objective, n_trials=25)
    # Reporting best trial and making a quick plot to examine hyperparameters
    trial = study.best_trial
    print(f"Best hyperparams: {trial.params}")
