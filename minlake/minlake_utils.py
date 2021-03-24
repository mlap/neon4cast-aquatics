import pandas as pd
import numpy as np
from diffu1D_u0 import *

df = pd.read_csv("minlake_test.csv", delimiter=",", index_col=0)
temp = df["groundwaterTempMean"].to_numpy()
par = df["uPARMean"].to_numpy()
do = df["dissolvedOxygen"].to_numpy()
chla = df["chlorophyll"].to_numpy()

def solve(params):
    """
    Solves the minlake model for the inputted BOD and k_r params inputted
    in dictionary form.
    """
    BOD = params["BOD"]
    k_r = params["k_r"]
    K_z = 2 * 10**(-5) # p.51
    a = K_z
    k_b = 0.1 # Table 5
    th_b = 1.047 # Table 5
    k_r = 0.1 # Table 5
    YCHO2 = 0.0083 # Table 5
    th_p = 1.036 # Table 5
    th_s = 1.065 # Table 5
    th_r = 1.047 # Table 5
    
    def P_max(n_t):
        return 9.6 * 1.036 **(temp[n_t] - 20) # Eq. 4

    def L_min(n_t):
        I = par[n_t]
        K_1 = 0.687 * 1.086**(temp[n_t] - 20)
        K_2 = 15
        return I * (1 + 2 * np.sqrt(K_1 / K_2)) / (I + K_1 + I**2 / K_2) # Eq. 5
    
    # f deals with sink and source terms 
    def f(x, n_t):
        return -1 / YCHO2 * k_r * th_r**(temp[n_t] - 20) * chla[n_t] + P_max(n_t) * L_min(n_t) * chla[n_t] - k_b * th_b**(temp[n_t]-20) * BOD 

    L = 3 # Length of domain
    dt = 1 # Mesh spacing in t
    F = a * dt # a * dt / dx**2
    T = 14 # Simulation time stop
    n_t_start = 0
    
    def I(x):
        return do[n_t_start]
        

    # Solving the PDE
    DO, x, t, _ = solver_FE_simple(I, a, f, L, dt, F, T)
    return DO, T
