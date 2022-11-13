import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from tqdm import tqdm


def baranyifun(t_arr, p):

    # Parameters
    mu_max, x0, xmax, lambd = p

    y0 = np.log(x0)
    ymax = np.log(xmax)

    A = t_arr + (1/mu_max) * np.log(np.exp(-mu_max * t_arr) + np.exp(-mu_max * lambd) - np.exp(-mu_max * t_arr -mu_max * lambd))

    y_t = y0 + mu_max * A - np.log(1 + (np.exp(mu_max * A) - 1) / (np.exp(ymax - y0)))
    x_t = np.exp(y_t)

    return x_t

def fsolvefun(pars, df, reactor_arr, fixed_xmax=True):
    length = len(reactor_arr)
    mu_maxs = pars[:length]
    lambdas = pars[length:2*length]
    xmaxs = pars[2*length:]
    x0 = 0.01

    error = 0
    for i in range(len(lambdas)):
        lambd = lambdas[i]
        mu_max = mu_maxs[i]
        if fixed_xmax:
            xmax = xmaxs[0]
        else:
            xmax = xmaxs[i]
        t_arr = df['Time'].iloc[:df[reactor_arr[i]].size]
        x_t = baranyifun(t_arr, [mu_max, x0, xmax, lambd])
        error += np.mean((x_t - df[reactor_arr[i]]) ** 2)

    if fixed_xmax:
        return error / len(lambdas) + np.mean((xmaxs - 10) ** 2) / 100
    else:
        return error / len(lambdas) + np.std(xmaxs) 


def mu_fit(data_df, reactor_arr, fsolve_guess=None, fixed_xmax=True):
    print('fitting mus')
    if fsolve_guess is None:
        if fixed_xmax:
            fsolve_guess = np.hstack([[0.8] * len(reactor_arr), [1] * len(reactor_arr), [10]])
        else:
            fsolve_guess = np.hstack([[0.8] * len(reactor_arr), [1] * len(reactor_arr), [10] * len(reactor_arr)])
        
    bounds = tuple([(1e-10, None)] * fsolve_guess.size)
    fsolve_sol = minimize(fsolvefun, fsolve_guess, args=(data_df, reactor_arr, fixed_xmax), method='L-BFGS-B', bounds=bounds, options={'eps': 1e-11})
    return fsolve_sol.x[:len(reactor_arr)], fsolve_sol.x[len(reactor_arr):2*len(reactor_arr)], fsolve_sol.x[2*len(reactor_arr):], fsolve_sol

def mu_fit_sequential(data_df, reactor_arr, fsolve_guess=None):
    print('fitting mus sequentially')
    mu_out = []
    xmax_out = []
    lambd_out = []
    fsolve_out = []

    if fsolve_guess is None:
        fsolve_guess = np.array([0.8, 1, 10])
    bounds = tuple([(1e-10, None)] + [(0, None)] + [(1e-10, None)])

    for i in tqdm(range(len(reactor_arr))):
        fsolve_sol = minimize(fsolvefun, fsolve_guess, args=(data_df[['Time',reactor_arr[i]]], [reactor_arr[i]], True), method='L-BFGS-B', \
            bounds=bounds, options={'eps': 1e-9, 'maxls': 100, 'gtol': 1e-15, 'ftol': 1e-15})
        mu_out.append(fsolve_sol.x[0])
        xmax_out.append(fsolve_sol.x[2])
        lambd_out.append(fsolve_sol.x[1])
        fsolve_out.append(fsolve_sol)

    mu_out = np.array(mu_out)
    xmax_out = np.array(xmax_out)
    lambd_out = np.array(lambd_out)

    return mu_out, lambd_out, xmax_out, fsolve_out