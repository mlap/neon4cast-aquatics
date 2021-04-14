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


def main(filename_num):
    df = pd.read_csv("minlake_test.csv", delimiter=",", index_col=0)

    training_data = df[["groundwaterTempMean", "uPARMean",
                        "dissolvedOxygen", "chlorophyll"]].loc[:28]
    # Normalizing data to -1, 1 scale; this improves performance of neural nets
    scaler = MinMaxScaler(feature_range=(-1, 1))
    training_data_normalized = scaler.fit_transform(training_data)

    model = LSTM(input_size=4, hidden_layer_size=128,
                 fc_size=128, output_size=14)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)

    epochs = 10000
    train_window = 7

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

    print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

    torch.save(model, f"models/trash_model_dist_{filename_num}.pkl")


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
    print("Active Cuda Device: GPU ", torch.cuda.current_device())
    for i in range(1):
        main(i)
