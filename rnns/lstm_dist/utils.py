import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy
import yaml

def make_forecast(args, params_etcs, means, stds):
    """
    Saves a forecast csv to forecast[args.csv_name].csv
    """
    dates = pd.date_range(start = args.start_date, end = args.end_date )
    columns = ['time', 'siteID', 'statistic', 'forecast', 'data_assimilation', 'oxygen']
    if params_etcs["variable"] == "do":
      columns.append("temp")
    df_means = pd.DataFrame(columns = ['time', 'siteID', 'statistic', 'forecast', 'data_assimilation', 'oxygen', 'temp'])
    df_means['time'] = dates
    df_means['siteID'] = params_etcs["csv_name"][:4]
    df_means['statistic'] = 'mean'
    df_means['forecast'] = 1
    df_means['data_assimilation'] = 0
    if params_etcs["variable"] == "do":
      df_means['oxygen'] = means[:, 2]
    df_means['temp'] = means[:, 0]
    
    df_stds = deepcopy(df_means)
    df_stds['statistic'] = 'sd'
    if params_etcs["variable"] == "do":
      df_stds['oxygen'] = stds[:, 2]
    df_stds['temp'] = stds[:, 0]
    
    df = df_means.append(df_stds)
    df.to_csv(f'forecast{params_etcs["csv_name"][:4]}.csv', index=False)

def plot(evaluation_data, means, stds, args, params_etcs, start_idx, end_idx):
    """
    Saves a matplotlib plot of dissolved oxygen and water temperature (or just water temperature if thats inputted)
    
    """
    data_len = len(evaluation_data)
    if params_etcs["variable"] == "do":
        fig, axs = plt.subplots(2)
        axs[0].plot(
            np.linspace(1, data_len, data_len), evaluation_data[:, 2]
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
            evaluation_data[:, 0],
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
    elif params_etcs["variable"] == "wt":
        plt.plot(
            np.linspace(1, data_len, data_len),
            evaluation_data[:, 0],
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


def evaluate(evaluation_data_normalized, condition_seq, args, scaler, params_etcs, model):
    """
    Returns the mean and std at each time point in the prediction window
    """
    # Conditioning lstm cells
    model.cpu()
    means = np.array([])
    stds = np.array([])
    model.eval()
    model.init_hidden("cpu")
    for seq, _ in condition_seq:
        with torch.no_grad():
            dist = build_dist(model, torch.Tensor(seq))
    
    dim = seq[-1].shape[0]
    # Now making the predictions

    test_inputs = evaluation_data_normalized[: -args.predict_window]
    means = np.array([])
    stds = np.array([])
    model.eval()
    for i in range(args.predict_window):
        seq = torch.FloatTensor(test_inputs[-params_etcs["train_window"]:])
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


def get_variables(params_etcs):
    """
    This returns the variables to select from the dataframe according to
    """
    # Selecting the right variables 
    if params_etcs["variable"] == "wt":
        vars = ["groundwaterTempMean"]
    elif params_etcs["variable"] == "do" and params_etcs["csv_name"] == "BARC_data":
        vars = ["groundwaterTempMean", "uPARMean", "dissolvedOxygen", "chlorophyll"]
    elif params_etcs["variable"] == "do" and params_etcs["csv_name"] == "POSE_data":
        vars = ["groundwaterTempMean", "turbidity", "dissolvedOxygen", "chlorophyll"]
    
    return vars

def save_etcs(args, params):
    """
    Saves important features/file names used in training
    """
    with open(f"models/{args.model_name}.yaml", 'w') as file:
      params["variable"] = args.variable
      params["epochs"] = args.epochs
      params["csv_name"] = args.csv_name
      params["network"] = args.network
      documents = yaml.dump(params, file)
      
def load_etcs(model_name):
    """
    Yank the important features/file names used in training
    """
    with open(f"models/{model_name}.yaml") as file:
        etcs = yaml.load(file, Loader=yaml.FullLoader)
    return etcs
    

def train(training_data_normalized, params, args, device, save_flag):
    """
    Trains a model; saves the model along with important features/file names
    """
    # Accounting for the number of drivers used for water temp vs DO
    if args.variable == "wt":
        input_dim = 1
    else:
        input_dim = 4
    # Initializing the LSTM model and putting everything on the GPU
    nets = {"lstm": LSTM, "gru": GRU}
    model = nets[args.network.lower()](
        input_dim=input_dim,
        hidden_dim=params["hidden_dim"],
        output_dim=2*input_dim,
        n_layers=params["n_layers"],
        device=device
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
        model.init_hidden(device)
        for seq, targets in train_seq:
            optimizer.zero_grad()
            model.float()
            
            # Using WT or DO build accordingly
            dist = build_dist(model, seq)

            targets = targets.view(len(targets), -1).float()
            single_loss = -dist.log_prob(targets)
            # Detaching to avoid autograd errors
            model.detach_hidden()
            single_loss.backward()
            optimizer.step()

        if i % 100 == 1:
            print(f"epoch: {i:3} loss: {single_loss.item():10.8f}")
    print(f"epoch: {i:3} loss: {single_loss.item():10.10f}")
    
    if save_flag:
        torch.save(model, f"models/{args.model_name}.pkl")
        params["final_loss"] = single_loss.item()
        save_etcs(args, params)
    else:
        return model
    

def get_data(csv_name):
    """
    This function reads the csv, and linearly interpolates the missing data
    """
    df = pd.read_csv(f"{csv_name}.csv", delimiter=",", index_col=0)
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
    """
    This takes in the data and puts into a list with inputs and targets to
    train the model
    """
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
    This function builds a covariance matrix from variates and covariates.
    After problems with trying to learn the covariance matrix with non-zero
    cross-terms, I decided to make the assumption that the covariates are zero.
    """
    # I add a small non-zero term here to avoid occasional instances when variance is evaluated to be zero
    cov_mat = torch.diag(torch.abs(var) + 1e-10)
    return cov_mat


def build_dist(model, seq):
    """
    This passes the input sequence to the model and then builds a multivariate
    normal distribution from the output of the model
    """
    y_pred = model(seq).view(-1)
    mu = y_pred[:seq[-1].shape[0]]
    var = y_pred[seq[-1].shape[0]:]
    cov_matrix = build_cov_matrix(var)
    return MultivariateNormal(mu, cov_matrix)


class LSTM(nn.Module):
    """
    Creating an object to handle an LSTM model
    """
    def __init__(
        self, input_dim=1, hidden_dim=64, output_dim=1, n_layers=2, device="cpu"
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers)
        self.linear = nn.Linear(self.hidden_dim, output_dim)
        self.n_layers = n_layers
        self.init_hidden(device)


    def forward(self, input):
        lstm_out, self.hidden_cell = self.lstm(
            input.view(len(input), 1, -1).float(), self.hidden_cell
        )
        out = self.linear(lstm_out[-1].view(1, -1))
        return out[-1]
    
    def init_hidden(self, device):
        self.hidden_cell = (
            torch.zeros(self.n_layers, 1, self.hidden_dim, device=device),
            torch.zeros(self.n_layers, 1, self.hidden_dim, device=device),
        )
    
    def detach_hidden(self):
        self.hidden_cell = (
                    self.hidden_cell[0].detach(),
                    self.hidden_cell[1].detach(),
                )
        
      
class GRU(nn.Module):
    """
    Creating an object to handle a GRU model
    """
    def __init__(
        self, input_dim=1, hidden_dim=64, output_dim=1, n_layers=2, device="cpu"
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, self.hidden_dim, n_layers)
        self.linear = nn.Linear(self.hidden_dim, output_dim)
        self.n_layers = n_layers
        self.init_hidden(device)
            

    def forward(self, input):
        gru_out, self.hidden_cell = self.gru(
            input.view(len(input), 1, -1).float(), self.hidden_cell
        )
        out = self.linear(gru_out[-1].view(1, -1))
        return out[-1]
    
    def init_hidden(self, device):
        self.hidden_cell = torch.zeros(self.n_layers, 1, self.hidden_dim, device=device)
    
    def detach_hidden(self):
        self.hidden_cell = self.hidden_cell.detach()
        
