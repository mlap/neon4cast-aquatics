import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from rnn_train_distribution import build_cov_matrix
from copy import deepcopy


def main():
    df = pd.read_csv("minlake_test.csv", delimiter=",", index_col=0)

    training_data = df[["groundwaterTempMean", "uPARMean",
                        "dissolvedOxygen", "chlorophyll"]].loc[:28]
    # Normalizing data to -1, 1 scale; this improves performance of neural nets
    scaler = MinMaxScaler(feature_range=(-1, 1))
    training_data_normalized = scaler.fit_transform(training_data)
    train_window = 7

    fut_pred = 7
    all_predictions = []
    for i in range(10):
        model = torch.load(f"models/trash_model_dist_0.pkl")
        test_inputs = training_data_normalized[-train_window -
                                               fut_pred:-fut_pred]
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
                dist = MultivariateNormal(mu, cov_matrix)
                samples = dist.rsample((10,))
                test_inputs = np.append(test_inputs, samples.mean(
                    axis=0).numpy()).reshape(-1, 4)

                means = np.append(means, scaler.inverse_transform(
                    samples).mean(axis=0)).reshape(-1, 4)
                stds = np.append(stds, scaler.inverse_transform(
                    samples).std(axis=0)).reshape(-1, 4)
        all_predictions.append(means)

    means = np.array(all_predictions).mean(axis=0)
    stds = np.array(all_predictions).std(axis=0)

    fig, axs = plt.subplots(2)
    axs[0].plot(np.linspace(1, 28, 28), training_data[["dissolvedOxygen"]])
    axs[0].errorbar(np.linspace(21, 28, 7), means[:, 2], stds[:, 2], capsize=5)
    axs[0].set_title("DO")
    axs[1].plot(np.linspace(1, 28, 28), training_data[["groundwaterTempMean"]])
    axs[1].errorbar(np.linspace(21, 28, 7), means[:, 0], stds[:, 0], capsize=5)
    axs[1].set_title("Groundwater Temp")
    plt.xlabel("Day")
    plt.savefig("trash.png")


if __name__ == "__main__":
    main()
