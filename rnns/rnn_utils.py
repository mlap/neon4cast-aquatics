import torch
import torch.nn as nn
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
        train_seq = input_data[i:i+train_window]
        train_target = input_data[i+train_window:i+train_window+1]
        seq.append((train_seq, train_target))

    return seq


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=500, fc_size=500, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.fc_0 = nn.Linear(hidden_layer_size, fc_size)
        self.fc_1 = nn.Linear(fc_size, fc_size)
        self.fc_2 = nn.Linear(fc_size, fc_size)
        self.fc_3 = nn.Linear(fc_size, fc_size)
        self.fc_4 = nn.Linear(fc_size, output_size)

        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(
            len(input_seq), 1, -1).float(), self.hidden_cell)
        out = self.relu(lstm_out)
        out = self.fc_0(out)
        out = self.relu(out)
        out = self.dropout(out)
        #out = self.fc_1(out)
        #out = self.relu(out)
        #out = self.dropout(out)
        #out = self.fc_2(out)
        #out = self.relu(out)
        #out = self.dropout(out)
        #out = self.fc_3(out)
        #out = self.relu(out)
        #out = self.dropout(out)
        out = self.fc_4(out.view(len(input_seq), -1))
        out = self.dropout(out)
        return out[-1]
