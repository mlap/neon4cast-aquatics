import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from rnn_utils import *
from copy import deepcopy
import argparse

# Argument parsing block; to get help on this from CL run `python tune_sb3.py -h`
parser = argparse.ArgumentParser()
parser.add_argument(
    "--png-name", type=str, default="trash_forecast", help="Name to save png"
)
parser.add_argument(
    "--model-name",
    type=str,
    default="64_lstm_64_hidden_BARC_final_0",
    help="Name of model to load",
)
parser.add_argument("--train-window", type=int, default=21)
parser.add_argument("--predict-window", type=int, default=7)
parser.add_argument("--start", type=int, default=-21)
args = parser.parse_args()


def main():
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df = pd.read_csv("minlake_test.csv", delimiter=",", index_col=0)
    df = df.sort_values(["year", "month", "day"])
    df = df.reset_index(drop=True)
    df['date'] = pd.to_datetime(df[["year", "month", "day"]])
    df.set_index('date', inplace=True)
    idx = pd.date_range(start = '2017-10-20', end = '2021-04-30' )
    df = df.reindex(idx, fill_value=np.NaN)
    df = df.interpolate(method ='linear', limit_direction ='forward')
    training_data = df[
        ["groundwaterTempMean", "uPARMean", "dissolvedOxygen", "chlorophyll"]
    ]
    # Normalizing data to -1, 1 scale; this improves performance of neural nets
    training_data_lstm = scaler.fit_transform(training_data)
    train_seq = create_sequence(
        training_data_lstm[: args.start], args.train_window
    )
    
    # Conditioning lstm cells
    model = torch.load(f"models/{args.model_name}.pkl")
    model.cpu()
    for i in range(1):
        means = np.array([])
        stds = np.array([])
        model.eval()
        model.hidden_cell = (
            torch.zeros(1, 1, model.hidden_layer_size),
            torch.zeros(1, 1, model.hidden_layer_size),
        )
        for seq, _ in train_seq:
            with torch.no_grad():
                dist = build_dist(model, torch.Tensor(seq))
    
    # Now making the predictions
    training_data = df[args.start :][
        ["groundwaterTempMean", "uPARMean", "dissolvedOxygen", "chlorophyll"]
    ]
    training_data_normalized = scaler.transform(training_data)

    for i in range(1):
        test_inputs = training_data_normalized
        means = np.array([])
        stds = np.array([])
        model.eval()
        for i in range(args.predict_window):
            seq = torch.FloatTensor(test_inputs[-args.train_window :])
            with torch.no_grad():
                dist = build_dist(model, seq)
                samples = dist.rsample((10000,))
                test_inputs = np.append(
                    test_inputs, samples.mean(axis=0).numpy()
                ).reshape(-1, 4)
                # import pdb; pdb.set_trace()
                scaled_samples = scaler.inverse_transform(samples)
                means = np.append(
                    means, np.mean(scaled_samples, axis=0)
                ).reshape(-1, 4)
                stds = np.append(stds, np.std(scaled_samples, axis=0)).reshape(
                    -1, 4
                )
                
    data_len = len(training_data_normalized)
    fig, axs = plt.subplots(2)
    axs[0].plot(
        np.linspace(1, data_len, data_len), training_data[["dissolvedOxygen"]]
    )
    axs[0].errorbar(
        np.linspace(
            data_len, data_len + args.predict_window, args.predict_window
        ),
        means[:, 2],
        stds[:, 2],
        capsize=5,
        marker="o",
    )
    axs[0].set_title("DO")
    axs[1].plot(
        np.linspace(1, data_len, data_len),
        training_data[["groundwaterTempMean"]],
    )
    axs[1].errorbar(
        np.linspace(
            data_len , data_len + args.predict_window, args.predict_window
        ),
        means[:, 0],
        stds[:, 0],
        capsize=5,
        marker="o",
    )
    axs[1].set_title("Groundwater Temp")
    plt.xlabel("Day")
    plt.savefig(f"{args.png_name}.png")


if __name__ == "__main__":
    main()
