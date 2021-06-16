import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy

def make_forecast(args, means, stds):
    """
    Saves a forecast csv to forecastSITE.csv
    """
    dates = pd.date_range(start = args.start_date, end = args.end_date )
    columns = ['time', 'siteID', 'statistic', 'forecast', 'data_assimilation', 'oxygen']
    if args.variable == "do":
      columns.append("temp")
    df_means = pd.DataFrame(columns = ['time', 'siteID', 'statistic', 'forecast', 'data_assimilation', 'oxygen', 'temp'])
    df_means['time'] = dates
    df_means['siteID'] = args.csv_name[:4]
    df_means['statistic'] = 'mean'
    df_means['forecast'] = 1
    df_means['data_assimilation'] = 0
    if args.variable == "do":
      df_means['oxygen'] = means[:, 2]
    df_means['temp'] = means[:, 0]
    
    df_stds = deepcopy(df_means)
    df_stds['statistic'] = 'sd'
    if args.variable == "do":
      df_stds['oxygen'] = stds[:, 2]
    df_stds['temp'] = stds[:, 0]
    
    df = df_means.append(df_stds)
    df.to_csv(f'forecast{args.csv_name[:4]}.csv', index=False)

def plot(evaluation_data, means, stds, args, start_idx, end_idx):
    data_len = len(evaluation_data)
    if args.variable == "do":
        fig, axs = plt.subplots(2)
        axs[0].plot(
            np.linspace(1, data_len, data_len), evaluation_data[["dissolvedOxygen"]]
        )
        axs[0].errorbar(
            np.linspace(
                start_idx, end_idx, args.predict_window
            ),
            means[:, 2],
            stds[:, 2],
            capsize=5,
            marker="o",
        )
        axs[0].set_title("DO")
        axs[1].plot(
            np.linspace(1, data_len, data_len),
            evaluation_data[["groundwaterTempMean"]],
        )
        axs[1].errorbar(
            np.linspace(
                start_idx, end_idx, args.predict_window
            ),
            means[:, 0],
            stds[:, 0],
            capsize=5,
            marker="o",
        )
        axs[1].set_title("Groundwater Temp")
        plt.xlabel("Day")
        plt.savefig(f"{args.png_name}.png")
    elif args.variable == "wt":
        plt.plot(
            np.linspace(1, data_len, data_len),
            evaluation_data[["groundwaterTempMean"]],
        )
        plt.errorbar(
            np.linspace(
                start_idx, end_idx, args.predict_window
            ),
            means[:, 0],
            stds[:, 0],
            capsize=5,
            marker="o",
        )
        plt.suptitle("Groundwater Temp")
        plt.xlabel("Day")
        plt.savefig(f"{args.png_name}.png")


def evaluate(evaluation_data, condition_seq, args, scaler):
    # Conditioning lstm cells
    model = torch.load(f"models/{args.model_name}.pkl")
    model.cpu()
    for i in range(1):
        means = np.array([])
        stds = np.array([])
        model.eval()
        model.hidden_cell = (
            torch.zeros(args.n_layers, 1, model.hidden_dim),
            torch.zeros(args.n_layers, 1, model.hidden_dim),
        )
        for seq, _ in condition_seq:
            with torch.no_grad():
                dist = build_dist(model, torch.Tensor(seq))
    
    dim = seq[-1].shape[0]
    # Now making the predictions
    evaluation_data_normalized = scaler.transform(evaluation_data)

    for i in range(1):
        test_inputs = evaluation_data_normalized[: -args.predict_window]
        means = np.array([])
        stds = np.array([])
        model.eval()
        for i in range(args.predict_window):
            seq = torch.FloatTensor(test_inputs[-args.train_window :])
            with torch.no_grad():
                dist = build_dist(model, seq)
                samples = dist.rsample((1000,))
                test_inputs = np.append(
                    test_inputs, samples.mean(axis=0).numpy()
                ).reshape(-1, dim)
                scaled_samples = scaler.inverse_transform(samples)
                means = np.append(
                    means, np.mean(scaled_samples, axis=0)
                ).reshape(-1, dim)
                stds = np.append(stds, np.std(scaled_samples, axis=0)).reshape(
                    -1, dim
                )
      
    return means, stds


def get_variables(args):
    # Selecting the right variables 
    if args.variable == "wt":
        vars = ["groundwaterTempMean"]
    elif args.variable == "do" and args.csv_name == "BARC_data.csv":
        vars = ["groundwaterTempMean", "uPARMean", "dissolvedOxygen", "chlorophyll"]
    elif args.variable == "do" and args.csv_name == "POSE_data.csv":
        vars = ["groundwaterTempMean", "turbidity", "dissolvedOxygen", "chlorophyll"]
    
    return vars

def train(training_data_normalized, params, args, device):
    # Accounting for the number of drivers used for water temp vs DO
    if args.variable == "wt":
        input_dim = 1
    else:
        input_dim = 4
    # Initializing the LSTM model and putting everything on the GPU    
    model = LSTM(
        input_dim=input_dim,
        hidden_dim=params["hidden_dim"],
        output_dim=2*input_dim,
    )
    model = model.to(device)
    training_data_normalized = torch.from_numpy(training_data_normalized).to(
        device
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["learning_rate"]
    )

    train_seq = create_sequence(
        training_data_normalized, params["train_window"]
    )
    # The training loop
    for i in range(args.epochs):
        model.hidden_cell = (
            torch.zeros(params["n_layers"], 1, model.hidden_dim).to(device),
            torch.zeros(params["n_layers"], 1, model.hidden_dim).to(device),
        )
        for seq, targets in train_seq:
            optimizer.zero_grad()
            model.float()
            
            # Using WT or DO build accordingly
            dist = build_dist(model, seq)

            targets = targets.view(len(targets), -1).float()
            single_loss = -dist.log_prob(targets)
            # Detaching to avoid autograd errors
            model.hidden_cell = (
                model.hidden_cell[0].detach(),
                model.hidden_cell[1].detach(),
            )
            single_loss.backward()
            optimizer.step()

        if i % 100 == 1:
            print(f"epoch: {i:3} loss: {single_loss.item():10.8f}")

    print(f"epoch: {i:3} loss: {single_loss.item():10.10f}")

    torch.save(model, f"models/{args.file_name}.pkl")

def get_data(csv_name):
    """
    This function reads the csv, and linearly interpolates the missing data
    """
    df = pd.read_csv(csv_name, delimiter=",", index_col=0)
    df = df.sort_values(["year", "month", "day"])
    df = df.reset_index(drop=True)
    df['date'] = pd.to_datetime(df[["year", "month", "day"]])
    # Collecting the first and last dates 
    start_date = str(df.iloc[0]["date"])[:10]
    end_date = str(df.iloc[-1]["date"])[:10]
    df.set_index('date', inplace=True)
    idx = pd.date_range(start = start_date, end = end_date)
    df = df.reindex(idx, fill_value=np.NaN)
    df = df.interpolate(method ='linear', limit_direction ='forward')
    
    return df

def create_sequence(input_data, train_window):
    seq = []
    L = len(input_data)
    # Loop splits up the data, so you have a slice of the train window and the
    # next item.
    for i in range(L - train_window):
        train_seq = input_data[i : i + train_window]
        train_target = input_data[i + train_window : i + train_window + 1]
        seq.append((train_seq, train_target))

    return seq


def build_cov_matrix(var):
    """
    This function builds a covariance matrix from variates and covariates
    """
    cov_mat = torch.diag(torch.abs(var) + 1e-10)
    return cov_mat


def build_dist(model, seq):
    y_pred = model(seq).view(-1)
    mu = y_pred[:seq[-1].shape[0]]
    var = y_pred[seq[-1].shape[0]:]
    cov_matrix = build_cov_matrix(var)
    return MultivariateNormal(mu, cov_matrix)


class LSTM(nn.Module):
    def __init__(
        self, input_dim=1, hidden_dim=64, output_dim=1, n_layers=2
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers)
        self.linear = nn.Linear(self.hidden_dim, output_dim)

        self.hidden_cell = (
            torch.zeros(n_layers, 1, self.hidden_dim),
            torch.zeros(n_layers, 1, self.hidden_dim),
        )

    def forward(self, input):
        lstm_out, self.hidden_cell = self.lstm(
            input.view(len(input), 1, -1).float(), self.hidden_cell
        )
        out = self.linear(lstm_out[-1].view(1, -1))
        return out[-1]
