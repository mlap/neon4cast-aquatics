import numpy as np
import pandas as pd
from diffu1D_u0 import *
import optuna
import matplotlib.pyplot as plt
from minlake_utils import *

def evaluate(study):
    """
    Defining the objective which will be some mean squared error from the model and NEON data
    Equation numbers etc. refer to Stefan and Fang 1994
    """
    best_trial = study.best_trial
    # Optuna trial is optimizing for BOD and k_r
    params= {"BOD": best_trial.params["BOD"],
             "k_r": best_trial.params["k_r"],
             }
    
    DO, T = solve(params)
    DO_data = np.array([[do[t] for i in range(4)] for t in range(T)])
    import pdb; pdb.set_trace()
    plt.plot(np.linspace(1, T, T), np.array(DO)[:,1:-1].mean(axis=1), label="prediction")
    #plt.plot(np.linspace(1, T, T), np.array(DO_data).mean(axis=1), label="data")
    plt.legend()
    plt.savefig("trash_fig")

if __name__ == "__main__":
    # Creating an Optuna study that uses sqlite
    study_name = "test"  # Unique identifier of the study.
    storage_name = "sqlite:///tuning_studies/{}.db".format(study_name)
    loaded_study = optuna.load_study(study_name=study_name, storage=storage_name)
    evaluate(loaded_study)


