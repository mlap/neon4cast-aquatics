import numpy as np
import pandas as pd
from diffu1D_u0 import *
import optuna
import matplotlib.pyplot as plt
from minlake_utils import *

def objective(trial):
    """
    Defining the objective which will be some mean squared error from the model and NEON data
    Equation numbers etc. refer to Stefan and Fang 1994
    """
    # Optuna trial is optimizing for BOD and k_r
    params= {"BOD": trial.suggest_uniform("BOD", 0, 1e3), #Review ranges here
             "k_r": trial.suggest_uniform("k_r", 0, 1), #Review Ranges here 
             }
    DO, T = solve(params)
    import pdb; pdb.set_trace()
    # Creating some bogus targets while database errors are happening
    DO_data = np.array([[do[t] for i in range(2)] for t in range(T)])

    # Using mean squared error as the measure of fit, where we want
    # to minimize this number
    return ((np.array(DO)[:,1:-1] - DO_data)**2).mean()

if __name__ == "__main__":
    # Creating an Optuna study that uses sqlite
    study_name = "test"  # Unique identifier of the study.
    storage_name = "sqlite:///tuning_studies/{}.db".format(study_name)
    # Sampling from hyperparameters using TPE over 1000 trials
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(study_name=study_name, sampler=sampler, direction='minimize', storage=storage_name, load_if_exists=True)
    study.optimize(objective, n_trials=1000)
    # Reporting best trial and making a quick plot to examine hyperparameters
    trial = study.best_trial
    print(f"Best hyperparams: {trial.params} \nBest Value: {trial.value}")


