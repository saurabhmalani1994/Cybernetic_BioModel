import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from tqdm import tqdm

def baranyifun_ode(t, var, par):
    X, S = var
    q0, mu_max, Y, Ks, m = par
    # Ks = 0.2
    m = mu_max

    a_t = q0 / (q0 + np.exp(-m * t))
    mu = mu_max * a_t * S / (Ks + S)

    dSdt = -mu * X / Y
    dXdt = mu * X

    return [dXdt, dSdt]


def baranyifun(t_arr, p):

    # Parameters
    x0, s0, q0, mu_max, Y, Ks, m = p
    tspan = [t_arr[0], t_arr[-1]]
    init = [x0, s0]
    pars = (q0, mu_max, Y, Ks, m,)
    sol = solve_ivp(baranyifun_ode, tspan, init, args=(pars,), t_eval=t_arr, method='BDF', atol=1e-6, rtol=1e-9)

    return sol.y[0,:]

def fsolvefun(pars, df, reactor_arr):
    q0, mu_max, Y, Ks, m = pars
    x0 = 0.01
    s0 = 2
    pars = x0, s0, q0, mu_max, Y, Ks, m

    error = 0
    t_arr = df['Time'].iloc[:df[reactor_arr[0]].size].to_numpy()
    x_t = baranyifun(t_arr, pars)
    error += np.mean((x_t - df[reactor_arr[0]]) ** 2)

    return error# + Ks / 100

def mu_fit_sequential(data_df, reactor_arr, fsolve_guess=None):
    print('fitting mus sequentially')
    

    if fsolve_guess is None:
        fsolve_guess = np.array([0.1, 0.1, 6, 0.01, 1])
    bounds = tuple([(1e-10, None)] + [(1e-10, None)] + [(1e-10, None)] + [(1e-10, None)] + [(1e-10, None)])

    append_sols = []

    for i in tqdm(range(len(reactor_arr))):
        fsolve_sol = minimize(fsolvefun, fsolve_guess, args=(data_df[['Time',reactor_arr[i]]], [reactor_arr[i]]), method='L-BFGS-B', \
            bounds=bounds, options={'eps': 1e-9, 'maxls': 100, 'gtol': 1e-15, 'ftol': 1e-15})
        append_sols.append(fsolve_sol)
        fsolve_guess = fsolve_sol.x

    return append_sols