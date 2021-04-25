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
parser.add_argument("--file-name", type=str,
                    default="trash_model_dist", help="Name to save model")
parser.add_argument("--epochs", type=int, default=25, help="Number of Epochs")
args=parser.parse_args()

params = {'learning_rate': 0.00001, 
          'train_window': 21, 
          'lstm_width': 512, 
          'hidden_width': 64}

def main(filename_num, device):
    df=pd.read_csv("minlake_test.csv", delimiter=",", index_col=0)
    df = df[df.year>2019].sort_values(['year', 'month', 'day'])
    training_data = df[["groundwaterTempMean", "uPARMean",
                        "dissolvedOxygen", "chlorophyll"]]
    # Normalizing data to -1, 1 scale; this improves performance of neural nets
    scaler=MinMaxScaler(feature_range=(-1, 1))
    training_data_normalized=scaler.fit_transform(training_data)
    
    model=LSTM(input_size=4, hidden_layer_size=params['lstm_width'],
                 fc_size=params['hidden_width'], output_size=14)
    model = model.to(device)
    import pdb; pdb.set_trace()
    training_data_normalized = torch.from_numpy(training_data_normalized).to(device)
    optimizer=torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    epochs=args.epochs

    train_seq=create_sequence(training_data_normalized, params['train_window'])
    for i in range(epochs):
        model.hidden_cell=(torch.zeros(1, 1, model.hidden_layer_size).to(device),
                                     torch.zeros(1, 1, model.hidden_layer_size).to(device))
        for seq, targets in train_seq:
            optimizer.zero_grad()
            model.float()
            dist=build_dist(model, seq)

            targets=targets.view(len(targets), -1).float()
            single_loss=-dist.log_prob(targets)
            # Detaching to avoid autograd errors
            model.hidden_cell=(
                model.hidden_cell[0].detach(), model.hidden_cell[1].detach())
            single_loss.backward()
            optimizer.step()

        if i % 100 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

    print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

    torch.save(model, f"models/{args.file_name}_{filename_num}.pkl")



if __name__ == "__main__":
    torch.cuda.set_device(0)
    print("Active Cuda Device: GPU ", torch.cuda.current_device())
    for i in range(1):
        main(i, torch.cuda.current_device())
