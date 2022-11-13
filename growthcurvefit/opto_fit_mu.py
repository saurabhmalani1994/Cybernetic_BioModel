import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import solve_ivp


def growthfun(t,x,p):

    # Parameters
    mu_max, Ks, Y = p

    # Variables
    S, X = x
    
    
    mu = mu_max * S / (Ks + S + 1e-10)

    dSdt = - (mu / (Y + 1e-10)) * X
    dXdt = mu * X

    return [dSdt, dXdt]


def fsolvefun(pars, df, reactor_dict, params_dict):
    # print(pars)

    X0 = pars[0]
    Y = pars[-1]
    Ks = pars[-2]
    mu_s = pars[1:-2]

    error = 0
    xinit = [2, X0]
    tspan = [df['Time'].iloc[0], df['Time'].iloc[-1]]

    # print(pars)

    for exp, light in reactor_dict.items():
        mu_max = mu_s[params_dict[light[0]]]
        pars_input = mu_max, Ks, Y
        sol = solve_ivp(growthfun, tspan, xinit, args=(pars_input,), method='BDF', t_eval=df['Time'], rtol=1e-6, atol=1e-9)
        error += np.sqrt(np.mean((df[exp] - sol.y[1, :]) ** 2))
    return error

def mu_fit(data_df, light_df, light_fit_first):
    # print('start')
    reactor_dict = light_df.to_dict('list')
    unique_lights = light_df.nunique(axis=1)[0]

    pars = np.zeros(unique_lights + 3) + 0.1
    # pars[0] = 1e-5

    index = 0
    params_dict = {}
    for key, value in reactor_dict.items():
        if value[0] not in params_dict:
            params_dict[value[0]] = index
            index += 1

    # Fit to one trajectory first.

    par_init = pars[[0,params_dict[light_fit_first],-2,-1]]
    # par_init = pars

    reactor_dict_init = {key: value for key, value in reactor_dict.items() if value[0] == light_fit_first}
    params_dict_init = {key: 0 for key, value in params_dict.items() if key == light_fit_first}
    bounds=tuple([(0, None)] * (3) + [(0, None)])

    fsolve_sol = minimize(fsolvefun, par_init, args=(data_df, reactor_dict_init, params_dict_init), method='L-BFGS-B', bounds=bounds, options={'ftol': 1e-11})  #options={'eps': 1e-11})

    print('initial done')
    print(fsolve_sol)

    pars[[0,params_dict[light_fit_first],-2,-1]] = fsolve_sol.x

    bounds=tuple([(0, None)] * (unique_lights + 3))
    fsolve_sol = minimize(fsolvefun, pars, args=(data_df, reactor_dict, params_dict), method='L-BFGS-B', bounds=bounds, options={'ftol': 1e-11})  #options={'eps': 1e-11})

    return fsolve_sol.x