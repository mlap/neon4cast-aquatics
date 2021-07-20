import torch
from sklearn.preprocessing import MinMaxScaler
from utils import *
import argparse

# Argument parsing block; to get help on this from CL run `python tune_sb3.py -h`
parser = argparse.ArgumentParser()
parser.add_argument(
    "--png-name",
    type=str,
    default="trash",
    help="Name of the png that will be saved in `rnns/`",
)
parser.add_argument(
    "--model-name",
    type=str,
    default="trash_model",
    help="Name of model to load from `models/`",
)
parser.add_argument(
    "--start",
    type=int,
    default=-29,
    help="Where to start the forecast, (-(how many places from the end))",
)
parser.add_argument("--predictive-net", action="store_true")
args = parser.parse_args()

def get_scaled_data(predictive_net_flag, params_etcs, df):
    """
    This function returns the appropriate data for the predictive net or AE.
    """
    if predictive_net_flag:
        util_dict = {"get_variables": get_variables_an}
    else:
        util_dict = {"get_variables": get_variables_ae}
    variables = util_dict["get_variables"](params_etcs)
    training_data = df[variables]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(training_data)
    return scaler, scaled_data

def main():
    params_etcs = load_etcs(args.model_name)
    df = get_data(params_etcs["csv_name"])
    scaler_ae, scaled_data_ae = get_scaled_data(False, params_etcs, df)
    scaler_pn, scaled_data_pn = get_scaled_data(True, params_etcs, df)
    # Creating sequences to condition the LSTM cells
    condition_seq_ae = create_sequence(
        scaled_data_ae[: args.start], params_etcs["train_window"], 1
    )
    condition_seq_pn = create_sequence(
        scaled_data_pn[: args.start], params_etcs["train_window"], 1
    )
    condition_seqs = (condition_seq_ae, condition_seq_pn)
    # Indexing the appropriate data
    end = args.start + params_etcs["train_window"] + params_etcs["prediction_window"]
    if end >= 0:
        end = None
    evaluation_data_ae = scaled_data_ae[args.start : end]
    evaluation_data_pn = scaled_data_pn[args.start : end]
    evaluation_data = (evaluation_data_ae, evaluation_data_pn)
    
    # Evaluating the data
    model_ae = torch.load(f"models/{args.model_name}_ae.pkl")
    model_pn = torch.load(f"models/{args.model_name}_pn.pkl")
    models = (model_ae, model_pn)
    means, stds = evaluate(
        evaluation_data, condition_seqs, args, scaler_pn, params_etcs, models
    )
    # Plotting the data
    data_len = len(evaluation_data)
    start_idx = data_len - params_etcs["prediction_window"] + 1
    end_idx = data_len
    plot(scaler.inverse_transform(evaluation_data), means, stds, args, params_etcs, start_idx, end_idx)


if __name__ == "__main__":
    main()
