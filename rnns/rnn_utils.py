import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy

# This function preprocesses data to create a training sequences and corresponding
# DO targets


def create_sequence(input_data, train_window):
    seq = []
    L = len(input_data)
    for i in range(L - train_window):
        train_seq = input_data[i : i + train_window]
        train_target = input_data[i + train_window : i + train_window + 1]
        seq.append((train_seq, train_target))

    return seq


def build_cov_matrix(var, covs):
    """
    This function builds a covariance matrix from variates and covariates
    """
    cov_mat = torch.diag(var)
    # cov_mat[0, 1] = 0
    # cov_mat[1, 0] = covs[0]
    # cov_mat[0, 2] = 0
    # cov_mat[2, 0] = covs[1]
    # cov_mat[0, 3] = 0
    # cov_mat[3, 0] = covs[2]
    # cov_mat[1, 2] = 0
    # cov_mat[2, 1] = covs[3]
    # cov_mat[1, 3] = 0
    # cov_mat[3, 1] = covs[4]
    # cov_mat[2, 3] = 0
    # cov_mat[3, 2] = covs[5]
    ## Enforcing matrix to be PD
    # cov_mat = torch.mm(cov_mat, torch.transpose(cov_mat, 0, 1))
    # cov_mat = cov_mat #+ torch.eye(len(cov_mat))
    return cov_mat


def build_dist(model, seq):
    y_pred = model(seq).view(-1)
    mu = y_pred[:4]
    var = torch.abs(y_pred[4:8])
    covs = y_pred[-6:]
    cov_matrix = build_cov_matrix(var, covs)
    return MultivariateNormal(mu, cov_matrix)


class LSTM(nn.Module):
    def __init__(
        self, input_size=1, hidden_layer_size=500, fc_size=500, output_size=1
    ):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.fc_0 = nn.Linear(hidden_layer_size, fc_size)
        self.fc_1 = nn.Linear(fc_size, fc_size)
        self.fc_2 = nn.Linear(fc_size, output_size)

        self.hidden_cell = (
            torch.zeros(1, 1, self.hidden_layer_size),
            torch.zeros(1, 1, self.hidden_layer_size),
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(
            input_seq.view(len(input_seq), 1, -1).float(), self.hidden_cell
        )
        out = self.relu(lstm_out)
        out = self.fc_0(out)
        out = self.relu(out)
        # out = self.dropout(out)
        out = self.fc_1(out)
        out = self.relu(out)
        # out = self.dropout(out)
        out = self.fc_2(out.view(len(input_seq), -1))
        # out = self.dropout(out)
        return out[-1]
