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
    #df = df[df.year>2019].sort_values(['year', 'month', 'day'])
    #training_data = df[["groundwaterTempMean", "uPARMean",
    #                    "dissolvedOxygen", "chlorophyll"]]
    training_data = df[["groundwaterTempMean", "uPARMean",
                        "dissolvedOxygen", "chlorophyll"]].loc[:8]
    # Normalizing data to -1, 1 scale; this improves performance of neural nets
    scaler = MinMaxScaler(feature_range=(-1, 1))
    training_data_normalized = scaler.fit_transform(training_data)
    model = LSTM(input_size=4, hidden_layer_size=7,
                 fc_size=56, output_size=14)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00007)

    epochs = 100
    train_window = 7

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
            model.hidden_cell = (model.hidden_cell[0].detach(), model.hidden_cell[1].detach())
            single_loss.backward()
            optimizer.step()

        if i % 25 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

    print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

    torch.save(model, f"models/trash_model_dist_{filename_num}.pkl")



if __name__ == "__main__":
    torch.cuda.set_device(0)
    print("Active Cuda Device: GPU ", torch.cuda.current_device())
    for i in range(1):
        main(i)
