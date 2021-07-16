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
    "--network", type=str, default="lstm_ae", help="Type of recurrent net to use"
)
parser.add_argument("--additional-net", action="store_true")
args = parser.parse_args()

# Edit hyperparameters here
params = {
    "learning_rate": 0.000001,
    "train_window": 1,
    "hidden_dim": 100,
    "embed_dim": 7,
    "n_layers": 3,
    "dropout": 0.3,
    "prediction_window": 7,
}

def get_scaled_data(additional_net_flag, params_etcs, df):
    if additional_net_flag:
        util_dict = {"get_variables": get_variables_an}
    else:
        util_dict = {"get_variables": get_variables_ae}
    variables = util_dict["get_variables"](params_etcs)
    training_data = df[variables]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(training_data)
    return scaler, scaled_data

def main(device):
    df = get_data(args.csv_name)
    params_etcs = {"variable": args.variable, "csv_name": args.csv_name, "addition_net": args.additional_net}
    scaler_ae, scaled_data_ae = get_scaled_data(False, params_etcs, df)
    # Training the model
    train_ae(scaled_data_ae, params, args, device, save_flag=True)
    if args.additional_net:
        scaler_an, scaled_data_an = get_scaled_data(args.additional_net, params_etcs, df)
        train_additional_net(scaled_data_an, scaled_data_ae, params, args, device, save_flag=True)


if __name__ == "__main__":
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.cuda.set_device(0)
    print("Active Cuda Device: GPU ", torch.cuda.current_device())
    main(torch.cuda.current_device())
