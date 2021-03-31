import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy

def main():
  df = pd.read_csv("minlake_test.csv", delimiter=",", index_col=0)
  
  training_data = df[["groundwaterTempMean", "uPARMean", "dissolvedOxygen", "chlorophyll"]].loc[:28]
  # Normalizing data to -1, 1 scale; this improves performance of neural nets
  scaler = MinMaxScaler(feature_range=(-1, 1))
  training_data_normalized = scaler.fit_transform(training_data)
  train_window=7
  
  fut_pred = 7
  model = torch.load("trash_model.pkl")
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

if __name__ == "__main__":
  main()
