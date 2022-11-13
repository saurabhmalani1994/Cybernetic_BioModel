import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import solve_ivp


def baranyifun(t_arr, p):

    # Parameters
    mu_max, x0, xmax, lambd, nu = p

    h0 = mu_max * lambd
    y0 = np.log(x0)
    ymax = np.log(xmax)
    m = 1

    y_t = y0 + mu_max * t_arr + (1/mu_max) * np.log(np.exp(-nu*t_arr) + np.exp(-h0) - np.exp(-nu*t_arr - h0)) \
          - (1/m) * np.log(1 + (np.exp(m*mu_max*t_arr + (1/mu_max) * np.log(np.exp(-nu*t_arr) + np.exp(-h0) - np.exp(-nu*t_arr - h0))) - 1) / np.exp(m*(ymax - y0)))

    x_t = np.exp(y_t)

    return x_t

def fsolvefun(pars, df, reactor_arr, fixed_xmax=True):
    length = len(reactor_arr)
    mu_maxs = pars[:length]
    lambdas = pars[length:2*length]
    xmaxs = pars[2*length:3*length]
    nus = pars[3*length:]
    x0 = 0.01

    error = 0
    for i in range(len(lambdas)):
        lambd = lambdas[i]
        mu_max = mu_maxs[i]
        if fixed_xmax:
            xmax = xmaxs[0]
            nu = nus[0]
        else:
            xmax = xmaxs[i]
            nu = nus[i]
        t_arr = df['Time'].iloc[:df[reactor_arr[i]].size]
        x_t = baranyifun(t_arr, [mu_max, x0, xmax, lambd, nu])
        error += np.mean((x_t - df[reactor_arr[i]]) ** 2)

    if fixed_xmax:
        return error / len(lambdas)
    else:
        return error / len(lambdas) #+ np.std(xmaxs) 


def mu_fit(data_df, reactor_arr, fsolve_guess=None, fixed_xmax=True):
    print('fitting mus')
    if fsolve_guess is None:
        if fixed_xmax:
            fsolve_guess = np.hstack([[0.8] * len(reactor_arr), [1] * len(reactor_arr), [10, 0.8]])
        else:
            fsolve_guess = np.hstack([[0.8] * len(reactor_arr), [1] * len(reactor_arr), [10] * len(reactor_arr), [1] * len(reactor_arr)])
        
    bounds = tuple([(1e-10, None)] * fsolve_guess.size)
    fsolve_sol = minimize(fsolvefun, fsolve_guess, args=(data_df, reactor_arr, fixed_xmax), method='L-BFGS-B', bounds=bounds, options={'eps': 1e-11})

    if fixed_xmax:
        return fsolve_sol.x[:len(reactor_arr)], \
               fsolve_sol.x[len(reactor_arr):2*len(reactor_arr)], \
               fsolve_sol.x[-2], \
               fsolve_sol.x[-1]
    else:
        return fsolve_sol.x[:len(reactor_arr)], \
               fsolve_sol.x[len(reactor_arr):2*len(reactor_arr)], \
               fsolve_sol.x[2*len(reactor_arr):3*len(reactor_arr)], \
               fsolve_sol.x[3*len(reactor_arr):4*len(reactor_arr)]