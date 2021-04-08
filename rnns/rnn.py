import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy

df = pd.read_csv("minlake_test.csv", delimiter=",", index_col=0)

# This function preprocesses data to create a training sequences and corresponding
# DO targets
def create_sequence(input_data, train_window):
  seq = []
  L = len(input_data)
  for i in range(L- train_window):
    train_seq = input_data[i:i+train_window]
    train_target = input_data[i+train_window:i+train_window+1]
    seq.append((train_seq, train_target))
  
  return seq

training_data = df[["groundwaterTempMean", "uPARMean", "dissolvedOxygen", "chlorophyll"]].loc[:28]
# Normalizing data to -1, 1 scale; this improves performance of neural nets
scaler = MinMaxScaler(feature_range=(-1, 1))
training_data_normalized = scaler.fit_transform(training_data)

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1).float(), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

model = LSTM(input_size=4, output_size=4)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 150
train_window=7
train_seq = create_sequence(training_data_normalized, train_window)

for i in range(epochs):
    for seq, labels in train_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        model.float()
        seq = torch.from_numpy(seq)
        y_pred = model(seq).view(1, -1)
        labels = torch.from_numpy(labels).view(len(labels),-1).float()
        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i%25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

fut_pred = 7

test_inputs = training_data_normalized[-train_window-fut_pred:-fut_pred]
output = np.array([])
model.eval()
for i in range(fut_pred):
  seq = torch.FloatTensor(test_inputs[-train_window:])
  with torch.no_grad():
    model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                          torch.zeros(1, 1, model.hidden_layer_size))
    test_inputs = np.append(test_inputs, model(seq).numpy().reshape(1,-1)).reshape(-1, 4)

actual_predictions = scaler.inverse_transform(test_inputs[train_window:])
plt.plot(np.linspace(1, 28, 28), training_data[["dissolvedOxygen"]])
plt.plot(np.linspace(21, 28, 7), actual_predictions[:, 2])
plt.savefig("trash.png")
