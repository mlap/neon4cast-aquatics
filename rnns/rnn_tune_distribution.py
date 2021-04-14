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
parser.add_argument("--study-name", type=str, help="Study name")
parser.add_argument("--n-trials", type=int, default=int(25),
                    help="Number of tuning trials")
args = parser.parse_args()


def get_params(trial):
    params = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1),
        'train_window': trial.suggest_categorical('train_window', [1, 7, 14, 21]),
        'lstm_width': trial.suggest_categorical('lstm_width', [64, 128, 256, 512]),
        'hidden_width': trial.suggest_categorical('hidden_width', [64, 128, 256, 512]),
    }

    return params


def score_model(model, params):
    df = pd.read_csv("minlake_test.csv", delimiter=",", index_col=0)

    training_data = df[["groundwaterTempMean", "uPARMean",
                        "dissolvedOxygen", "chlorophyll"]].loc[:28]
    # Normalizing data to -1, 1 scale; this improves performance of neural nets
    scaler = MinMaxScaler(feature_range=(-1, 1))
    training_data_normalized = scaler.fit_transform(training_data)
    fut_pred = 7
    train_window = params['train_window']
    all_predictions = []
    test_inputs = training_data_normalized[-train_window -
                                           fut_pred:-fut_pred]
    for i in range(10):
        means = np.array([])
        stds = np.array([])
        model.eval()
        for i in range(fut_pred):
            seq = torch.FloatTensor(test_inputs[-train_window:])
            with torch.no_grad():
                model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                     torch.zeros(1, 1, model.hidden_layer_size))
                y_pred = model(seq)
                mu = y_pred[:4]
                var = torch.abs(y_pred[4:8])
                covs = y_pred[-6:]
                cov_matrix = build_cov_matrix(var, covs)
                #import pdb; pdb.set_trace()
                dist = MultivariateNormal(mu, cov_matrix)
                #import pdb; pdb.set_trace()
                samples = dist.rsample((10,))
                test_inputs = np.append(test_inputs, samples.mean(
                    axis=0).numpy()).reshape(-1, 4)

                means = np.append(means, scaler.inverse_transform(
                    samples).mean(axis=0)).reshape(-1, 4)

        all_predictions.append(means)
    means = np.array(all_predictions).mean(axis=0)
    stds = np.array(all_predictions).std(axis=0)

    means = np.array(all_predictions).mean(axis=0)
    stds = np.array(all_predictions).std(axis=0)
    DO_targets = training_data[["dissolvedOxygen"]
                               ][-fut_pred:].to_numpy().reshape(-1)
    WT_targets = training_data[["groundwaterTempMean"]
                               ][-fut_pred:].to_numpy().reshape(-1)
    objective = ((means[:, 2] - DO_targets)**2).mean() + ((means[:, 2] + stds[:, 2] - DO_targets)**2).mean() + \
        ((means[:, 0] - WT_targets)**2).mean() + \
        ((means[:, 0] + stds[:, 0] - WT_targets)**2).mean()

    return objective


def train_model(params):
    df = pd.read_csv("minlake_test.csv", delimiter=",", index_col=0)

    training_data = df[["groundwaterTempMean", "uPARMean",
                        "dissolvedOxygen", "chlorophyll"]].loc[:28]
    # Normalizing data to -1, 1 scale; this improves performance of neural nets
    scaler = MinMaxScaler(feature_range=(-1, 1))
    training_data_normalized = scaler.fit_transform(training_data)

    model = LSTM(
        input_size=4, hidden_layer_size=params['lstm_width'], fc_size=params['hidden_width'], output_size=14)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params['learning_rate'])

    epochs = 150
    train_window = params['train_window']

    train_seq = create_sequence(training_data_normalized, train_window)

    for i in range(epochs):
        for seq, targets in train_seq:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))
            model.float()
            seq = torch.from_numpy(seq)
            y_pred = model(seq).view(-1)
            mu = y_pred[:4]
            var = torch.abs(y_pred[4:8])
            covs = y_pred[-6:]
            cov_matrix = build_cov_matrix(var, covs)
            #import pdb; pdb.set_trace()
            dist = MultivariateNormal(mu, cov_matrix)

            targets = torch.from_numpy(targets).view(len(targets), -1).float()
            single_loss = -dist.log_prob(targets)
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


def build_cov_matrix(var, covs):
    """
    This function builds a covariance matrix from variates and covariates
    """
    cov_mat = torch.diag(var)
    cov_mat[0, 1] = 0
    cov_mat[1, 0] = covs[0]
    cov_mat[0, 2] = 0
    cov_mat[2, 0] = covs[1]
    cov_mat[0, 3] = 0
    cov_mat[3, 0] = covs[2]
    cov_mat[1, 2] = 0
    cov_mat[2, 1] = covs[3]
    cov_mat[1, 3] = 0
    cov_mat[3, 1] = covs[4]
    cov_mat[2, 3] = 0
    cov_mat[3, 2] = covs[5]
    # Enforcing matrix to be PD
    cov_mat = torch.mm(cov_mat, cov_mat.t())
    cov_mat.add_(torch.eye(len(cov_mat)))
    try:
        np.linalg.cholesky(cov_mat.detach().numpy())
        return cov_mat
    except:
        import pdb
        pdb.set_trace()


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
