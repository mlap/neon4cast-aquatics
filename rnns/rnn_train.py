import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy
from rnn_utils import *


def main(filename_num):
    df = pd.read_csv("minlake_test.csv", delimiter=",", index_col=0)

    training_data = df[
        ["groundwaterTempMean", "uPARMean", "dissolvedOxygen", "chlorophyll"]
    ].loc[:28]
    # Normalizing data to -1, 1 scale; this improves performance of neural nets
    scaler = MinMaxScaler(feature_range=(-1, 1))
    training_data_normalized = scaler.fit_transform(training_data)

    model = LSTM(input_size=4, output_size=4)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 300
    train_window = 7
    train_seq = create_sequence(training_data_normalized, train_window)

    for i in range(epochs):
        for seq, labels in train_seq:
            optimizer.zero_grad()
            model.hidden_cell = (
                torch.zeros(1, 1, model.hidden_layer_size),
                torch.zeros(1, 1, model.hidden_layer_size),
            )
            model.float()
            seq = torch.from_numpy(seq)
            y_pred = model(seq).view(1, -1)
            labels = torch.from_numpy(labels).view(len(labels), -1).float()
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        if i % 25 == 1:
            print(f"epoch: {i:3} loss: {single_loss.item():10.8f}")

    print(f"epoch: {i:3} loss: {single_loss.item():10.10f}")

    torch.save(model, f"models/trash_model{filename_num}.pkl")


if __name__ == "__main__":
    torch.cuda.set_device(0)
    print("Active Cuda Device: GPU ", torch.cuda.current_device())
    for i in range(10):
        main(i)
