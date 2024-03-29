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
    df_means = pd.DataFrame(columns = ['time', 'siteID', 'statistic', 'forecast', 'data_assimilation', 'oxygen', 'temperature'])
    df_means['time'] = dates
    df_means['siteID'] = params_etcs["csv_name"][:4]
    df_means['statistic'] = 'mean'
    df_means['forecast'] = 1
    df_means['data_assimilation'] = 0
    if params_etcs["variable"] == "do":
        df_means['oxygen'] = means[:, 2]
    df_means['temperature'] = means[:, 0]
    
    df_stds = deepcopy(df_means)
    df_stds['statistic'] = 'sd'
    if params_etcs["variable"] == "do":
        df_stds['oxygen'] = stds[:, 2]
    df_stds['temperature'] = stds[:, 0]
    
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
            np.linspace(1, data_len, data_len), evaluation_data[:, 1]
        )
        axs[0].errorbar(
            np.linspace(
                start_idx, end_idx, params_etcs["prediction_window"]
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
                start_idx, end_idx, params_etcs["prediction_window"]
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
                start_idx, end_idx, params_etcs["prediction_window"]
            ),
            means[:, 0],
            stds[:, 0],
            capsize=5,
            marker="o",
        )
        plt.suptitle("Groundwater Temp")
        plt.xlabel("Day")
        plt.savefig(f"{args.png_name}.png")


def evaluate(evaluation_data, condition_seqs, args, scaler, params_etcs, models):
    """
    Returns the mean and std at each time point in the prediction window
    """
    # Conditioning lstm cells
    # First initializing both models
    model_ae, model_pn = models
    model_ae.cpu()
    model_pn.cpu()
    means = np.array([])
    stds = np.array([])
    model_ae.init_hidden("cpu")
    model_pn.init_hidden("cpu")
    condition_seq_ae, condition_seq_pn = condition_seqs
    for i, e in enumerate(condition_seq_ae):
        with torch.no_grad():
            seq_ae, _ = e
            seq_pn, _ = condition_seq_pn[i]
            # Forward pass through autoencoder
            model_ae(torch.from_numpy(seq_ae))
            ae_embedding = model_ae.embedding
            # I want the same input sequence from the predictive network but the last item
            seq = torch.cat((torch.from_numpy(seq_pn[-1]), ae_embedding[0]), dim=0)
            # Forward pass through predictive net
            y_pred = model_pn(seq.reshape(1, -1)).view(-1)
    
    
    # Now making the predictions
    evaluation_data_ae, evaluation_data_pn = evaluation_data
    seq_ae = evaluation_data_ae[: -params_etcs["prediction_window"]]
    input_pn = evaluation_data_pn[-params_etcs["prediction_window"] - 1]
    dim = input_pn.shape[0]
    means = np.array([])
    stds = np.array([])
    with torch.no_grad():
        samples_final = np.array([])
        for day in range(params_etcs["prediction_window"]):
            samples = np.array([])
            samples_ae = np.array([])
            # Making sure the autoencoder keeps the correct hidden cell        
            encoder_hidden_cell = model_ae.hidden_cell_ae
            decoder_hidden_cell = model_ae.hidden_cell_de
            # Likewise with the predictive network
            predictive_hidden_cell = model_pn.hidden_cell
            # Collect multiple forward passes
            for i in range(100):
                # Resetting hidden cell every iteration
                model_ae.hidden_cell_ae = encoder_hidden_cell 
                model_ae.hidden_cell_de = decoder_hidden_cell
                model_pn.hidden_cell = predictive_hidden_cell
                # Forward pass on AE to get embedding
                ae_pred = model_ae(torch.from_numpy(seq_ae))
                ae_embedding = model_ae.embedding
                seq = torch.cat((torch.from_numpy(input_pn).reshape(-1), ae_embedding.reshape(-1)), dim=0).reshape(1, -1)
                # Doing forward pass and keeping track of past AE samples
                samples = np.append(samples, scaler.inverse_transform(model_pn(seq.reshape(1, -1)).numpy().reshape(-1, dim))).reshape(i+1, -1, dim)
                samples_ae = np.append(samples_ae, ae_pred).reshape(-1, 14)
            ae_pred = samples_ae.mean(axis=0)
            input_pn = scaler.transform(samples.mean(axis=0))
            seq_ae = np.append(evaluation_data_ae, ae_pred.reshape(-1,2)).reshape(-1, 2)[day+1:day+1+params_etcs["train_window"]]
            samples_final = np.append(samples_final, samples).reshape(day+1, -1, dim)
        means = samples_final.mean(axis=1)
        stds = samples_final.std(axis=1)
            
    return means, stds


def get_variables_an(params_etcs):
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

def get_variables_ae(params_etcs):
    """
    This returns the variables to select from the dataframe according to
    """
    # Selecting the right variables 
    if params_etcs["variable"] == "wt":
        vars = ["groundwaterTempMean"]
    elif params_etcs["variable"] == "do" and params_etcs["csv_name"] == "BARC_data":
        vars = ["groundwaterTempMean", "dissolvedOxygen"]
    elif params_etcs["variable"] == "do" and params_etcs["csv_name"] == "POSE_data":
        vars = ["groundwaterTempMean", "dissolvedOxygen"]
    
    return vars

def save_etcs(args, params):
    """
    Saves important features/file names used in training
    """
    with open(f"models/{args.model_name}_ae.yaml", 'w') as file:
      params["variable"] = args.variable
      params["epochs"] = args.epochs
      params["csv_name"] = args.csv_name
      params["network"] = args.network
      documents = yaml.dump(params, file)
      
def load_etcs(model_name):
    """
    Yank the important features/file names used in training
    """
    with open(f"models/{model_name}_ae.yaml") as file:
        etcs = yaml.load(file, Loader=yaml.FullLoader)
    return etcs
    

def train_autoencoder(training_data_normalized, params, args, device, save_flag):
    """
    Trains a model; saves the model along with important features/file names
    """
    # Accounting for the number of drivers used for water temp vs DO
    if args.variable == "wt":
        input_scale = 1
    else:
        # With the autoencoder I am just going to look at DO and WT
        input_scale = 2
    # Initializing the LSTM model and putting everything on the GPU    
    nets = {"lstm_ae": LSTMAutoEncoder, }
    model = nets[args.network.lower()](
        input_dim=input_scale,
        hidden_dim=params["hidden_dim"],
        embed_dim=params["embed_dim"],
        output_dim=input_scale * params["prediction_window"],
        n_layers=params["n_layers"],
        dropout=params["dropout"],
        device=device,
    )
    
    model = model.to(device)
    training_data_normalized = torch.from_numpy(training_data_normalized).to(
        device
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["learning_rate"]
    )

    train_seq = create_sequence(
        training_data_normalized, params["train_window"], params["prediction_window"]
    )
    # The training loop
    for i in range(args.epochs):
        model.init_hidden(device)
        for seq, targets in train_seq:
            optimizer.zero_grad()
            model.float()
            # Forward pass
            y_pred = model(seq).view(-1)
            targets = targets.reshape(-1).float()
            # Computing the loss
            loss = nn.MSELoss()
            output = loss(y_pred, targets)
            # Detaching to avoid autograd errors
            model.detach_hidden()
            # Gradient step
            output.backward()
            optimizer.step()

        if i % 10 == 1:
            print(f"AE epoch: {i:3} loss: {output.item():10.8f}")
    print(f"AE epoch: {i:3} loss: {output.item():10.10f}")
    
    if save_flag:
        torch.save(model, f"models/{args.model_name}_ae.pkl")
        params["final_ae_loss"] = output.item()
        save_etcs(args, params)
    else:
        return model

def train_predictive_net(scaled_data_an, scaled_data_ae, params, args, device, save_flag):
    # WIP
    model_ae = torch.load(f"models/{args.model_name}_ae.pkl")
    model_ae = model_ae.to(device)
    # Accounting for the number of drivers used for water temp vs DO
    if args.variable == "wt":
        input_dim = 1
    else:
        input_dim = 4
    model = GRU(
        input_dim=input_dim + model_ae.embed_dim,
        hidden_dim=params["hidden_dim"],
        output_dim=input_dim,
        n_layers=params["n_layers"],
        dropout=params["dropout"],
        device=device,
    )
    model = model.to(device)
    scaled_data_an = torch.from_numpy(scaled_data_an).to(
        device
    )
    scaled_data_ae = torch.from_numpy(scaled_data_ae).to(
        device
    )
    
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["learning_rate"]
    )
    # Try here again with longer prediction window
    train_seq = create_sequence(
        scaled_data_an, params["train_window"], 1
    )
    train_seq_ae = create_sequence(
        scaled_data_ae, params["train_window"], 1
    )
    # The training loop
    for i in range(args.epochs):
        model.init_hidden(device)
        model_ae.init_hidden(device)
        # Going need to get separate data variables for external drivers
        for j, e in enumerate(train_seq):
            seq_pn, targets = e
            seq_ae, _ = train_seq_ae[j]
            # Setting up for gradient descent
            optimizer.zero_grad()
            model.float()
            model_ae(seq_ae)
            ae_embedding = model_ae.embedding
            seq = torch.cat((seq_pn[-1], ae_embedding[0]), dim=0)
            # Forward pass
            y_pred = model(seq.reshape(1, -1)).view(-1)

            targets = targets.view(-1).float()
            # Computing the loss
            loss = nn.MSELoss()
            output = loss(y_pred, targets)
            # Detaching to avoid autograd errors
            model.detach_hidden()
            model_ae.detach_hidden()
            # Gradient step
            output.backward()
            optimizer.step()

        if i % 10 == 1:
            print(f"AN epoch: {i:3} loss: {output.item():10.8f}")
    print(f"AN epoch: {i:3} loss: {output.item():10.10f}")
    
    if save_flag:
        torch.save(model, f"models/{args.model_name}_pn.pkl")
        params["final_ae_loss"] = output.item()
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

def create_sequence(input_data, train_window, prediction_window):
    """
    This takes in the data and puts into a list with inputs and targets to
    train the model
    """
    seq = []
    L = len(input_data)
    # Loop splits up the data, so you have a slice of the train window and the
    # next item.
    for i in range(L - train_window - (prediction_window-1)):
        train_seq = input_data[i : i + train_window]
        train_target = input_data[i + train_window : i + train_window + prediction_window]
        seq.append((train_seq, train_target))

    return seq



class LSTMAutoEncoder(nn.Module):
    """
    Creating an object to handle an LSTM model
    """
    def __init__(
        self, input_dim=2, hidden_dim=64, output_dim=1, embed_dim=1, n_layers=2, dropout=0.1, device="cpu"
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        # Encoder
        self.encoder = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=dropout)
        self.linear_ae = nn.Linear(hidden_dim, embed_dim)
        self.embedding = None
        self.embed_dim = embed_dim
        # Decoder
        self.decoder = nn.LSTM(embed_dim, hidden_dim, n_layers, dropout=dropout)
        self.linear_de = nn.Linear(hidden_dim, output_dim)
        self.init_hidden(device)

    def forward(self, input):
        lstm_out, self.hidden_cell_ae = self.encoder(
            input.view(len(input), 1, -1).float(), self.hidden_cell_ae
        )
        self.embedding = self.linear_ae(lstm_out[-1].view(1, -1))
        lstm_out, self.hidden_cell_de = self.decoder(
            self.embedding.view(len(self.embedding), 1, -1).float(), self.hidden_cell_de
        )
        out = self.linear_de(lstm_out[-1].view(1, -1))
        return out[-1]
    
    def init_hidden(self, device):
        self.hidden_cell_ae = (
            torch.zeros(self.n_layers, 1, self.hidden_dim, device=device),
            torch.zeros(self.n_layers, 1, self.hidden_dim, device=device),
        )
        self.hidden_cell_de = (
            torch.zeros(self.n_layers, 1, self.hidden_dim, device=device),
            torch.zeros(self.n_layers, 1, self.hidden_dim, device=device),
        )
    
    def detach_hidden(self):
        self.hidden_cell_ae = (
                    self.hidden_cell_ae[0].detach(),
                    self.hidden_cell_ae[1].detach(),
                )
        self.hidden_cell_de = (
                    self.hidden_cell_de[0].detach(),
                    self.hidden_cell_de[1].detach(),
                )
      
class GRU(nn.Module):
    """
    Creating an object to handle a GRU model
    """
    def __init__(
        self, input_dim=1, hidden_dim=64, output_dim=1, n_layers=2, dropout=0.1, device="cpu"
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, self.hidden_dim, n_layers, dropout=dropout)
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
