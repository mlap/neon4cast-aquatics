import torch
from sklearn.preprocessing import MinMaxScaler
from utils import *
import argparse
import os

# Argument parsing block; to get help on this from CL run `python tune_sb3.py -h`
parser = argparse.ArgumentParser()
parser.add_argument(
    "--csv-name", type=str, default="POSE_data", help="Name of CSV to use"
)
parser.add_argument(
    "--model-name",
    type=str,
    default="trash_model",
    help="Name the model to be saved in `models/`",
)
parser.add_argument(
    "--epochs", type=int, default=1, help="Number of epochs to train for"
)
parser.add_argument(
    "--variable",
    type=str,
    default="do",
    help="Name of variable being predicted (wt - water temperature, do - water temperature and dissolved oxygen)",
)
parser.add_argument(
    "--network", type=str, default="lstm", help="Type of recurrent net to use"
)
args = parser.parse_args()

# Edit hyperparameters here
params = {
    "learning_rate": 0.000001,
    "train_window": 1,
    "hidden_dim": 64,
    "n_layers": 2,
    "dropout": 0.4,
}


def main(device):
    df = get_data(args.csv_name)
    params_etcs = {"variable": args.variable, "csv_name": args.csv_name}
    variables = get_variables(params_etcs)
    training_data = df[variables]
    # Normalizing data to -1, 1 scale; this improves performance of neural nets
    scaler = MinMaxScaler(feature_range=(-1, 1))
    training_data_normalized = scaler.fit_transform(training_data)
    # Training the model
    train(training_data_normalized, params, args, device, save_flag=True)


if __name__ == "__main__":
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.cuda.set_device(0)
    print("Active Cuda Device: GPU ", torch.cuda.current_device())
    main(torch.cuda.current_device())
