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
parser.add_argument(
    "--predict-window",
    type=int,
    default=7,
    help="How long of a forecast to make",
)
args = parser.parse_args()


def main():
    params_etcs = load_etcs(args.model_name)
    df = get_data(params_etcs["csv_name"])
    variables = get_variables(params_etcs)
    data = df[variables]
    # Normalizing data to -1, 1 scale; this improves performance of neural nets
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_scaled = scaler.fit_transform(data)
    # Creating a sequence to condition the LSTM cells
    condition_seq = create_sequence(
        data_scaled[: args.start], params_etcs["train_window"]
    )
    # Indexing the appropriate data
    end = args.start + params_etcs["train_window"] + args.predict_window
    evaluation_data = data_scaled[args.start : end]
    # Evaluating the data
    model = torch.load(f"models/{args.model_name}.pkl")
    means, stds = evaluate(
        evaluation_data, condition_seq, args, scaler, params_etcs, model
    )
    # Plotting the data
    data_len = len(evaluation_data)
    start_idx = data_len - args.predict_window
    end_idx = data_len
    plot(scaler.inverse_transform(evaluation_data), means, stds, args, params_etcs, start_idx, end_idx)


if __name__ == "__main__":
    main()
