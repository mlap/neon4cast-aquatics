import numpy as np
from diffu1D_u0 import *
import optuna

def objective(trial):
    """
    Defining the objective which will be some mean squared error from the model and NEON data
    Equation numbers etc. refer to Stefan and Fang 1993
    """
    # The parameters that we will calibrate the model for are shown here.
    # Optuna trial i
    BOD = trial.suggest_uniform("BOD", 0, 1) #Review ranges here
    k_r = trial.suggest_uniform("k_r", 0, 1) #Review Ranges here 
    
    def ChLa(t):
        return 1 # Need to link to data

    def I(x):
        return 1 # Need to link to data

    K_z = 2 * 10**(-5) # p.51
    a = K_z
    k_b = 0.1 # Table 5
    th_b = 1.047 # Table 5
    k_r = 0.1 # Table 5
    YCHO2 = 0.0083 # Table 5
    th_p = 1.036 # Table 5
    th_s = 1.065 # Table 5
    th_r = 1.047 # Table 5

    def Temp(t):
        """
        Function that maps time to temperature
        """
        return 20 # Need to link to data

    def P_max(t):
        return 9.6 * 1.036 **(Temp(t) - 20) # Eq. 4

    def L_min(t):
        I = 1 # Need to link to PAR data
        K_1 = 0.687 * 1.086**(Temp(t) - 20)
        K_2 = 15
        return I * (1 + 2 * np.sqrt(K_1 / K_2)) / (I + K_1 + I**2 / K_2) # Eq. 5
    
    # f deals with sink and source terms 
    def f(x, t):
        return -1 / YCHO2 * k_r * th_r**(Temp(t) - 20) * ChLa(t) + P_max(t) * L_min(t) * ChLa(t) - k_b * th_b**(Temp(t)-20) * BOD 

    L = 200 # Length of domain
    dt = 1 / 48 # Mesh spacing in t
    F = a * dt # a * dt / dx**2
    T = 100 # Simulation time stop

    # Solving the PDE
    DO, x, t, _ = solver_FE_simple(I, a, f, L, dt, F, T)
    
    # Creating some bogus targets while database errors are happening
    DO_data = DO + np.random.random(len(DO))

    # Using mean squared error as the measure of fit, where we want
    # to minimize this number
    return ((DO - DO_data)**2).mean()

if __name__ == "__main__":
    # Creating an Optuna study that uses sqlite
    study_name = "test"  # Unique identifier of the study.
    storage_name = "sqlite:///tuning_studies/{}.db".format(study_name)
    # Sampling from hyperparameters using TPE over 50 trials
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(study_name=study_name, sampler=sampler, direction='minimize', storage=storage_name, load_if_exists=True)
    study.optimize(objective, n_trials=50)
    # Reporting best trial and making a quick plot to examine hyperparameters
    trial = study.best_trial
    print(f"Best hyperparams: {trial.params}")


