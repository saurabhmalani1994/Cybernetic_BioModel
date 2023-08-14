import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import minimize, root
from scipy.integrate import solve_ivp
from tqdm import tqdm
import pandas as pd
from scipy.signal import savgol_filter
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

def get_pars(kE = 3.54996268e-02,
             kE_basal = 5.92764941e-04,
             nE =  2.56241667e+00,
             KE = 1.14559216e-02):

    return kE, kE_basal, nE, KE

def SGy139_ODE(t, x, parameters, pulse, time_delay=4.36486878e+00):
    if callable(pulse):
        pulse = pulse(t - time_delay)
    # Unpack Parameters
    kE, kE_basal, nE, KE = parameters

    mu = x
    dxdt = kE_basal + kE * (pulse ** nE) / (pulse ** nE + KE ** nE) - (mu) * x
    return dxdt

def pred_ss_fun(parameters, pulse_arr):
    pred_ss = []
    f = lambda x, parameters, pulse: SGy139_ODE(0, x, parameters, pulse)
    for pulse in pulse_arr:
        x_ss = root(f, 0.5, args=(parameters, pulse)).x
        pred_ss.append(x_ss)
    return np.array(pred_ss).flatten()

def makeLightDoseResponseCurve(pulse_arr=None, parameters=None, make_plot=True):
    if pulse_arr is None:
        pulse_arr = np.concatenate((np.array([0]), np.logspace(-3, -1, 100)))
    if parameters is None:
        parameters = get_pars()

    mu_arr = pred_ss_fun(parameters, pulse_arr)

    if make_plot:
        steady_state_output, _, _ = GroundTruthData()
        pulse_groundtruth, mu_groundtruth, _ = steady_state_output

        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        ax.plot(pulse_arr, mu_arr, '-', label="Model", color="blue", linewidth=2)
        ax.plot(pulse_groundtruth, mu_groundtruth, 'o', label="Data", color="black", linewidth=2)
        ax.set_xlabel("Light Pulsing Fraction", fontsize=14)
        ax.set_ylabel("Growth Rate (1/hr)", fontsize=14)
        ax.set_xscale("symlog", linthresh=0.01)
        ax.tick_params(axis='both', which='major', labelsize=14, width=2, length=5, direction='in')
        ax.tick_params(axis='x', which='minor', labelsize=14, width=1, length=3, direction='in')
        # ax.xaxis.set_minor_locator(MultipleLocator(0.002))
        # ax.xaxis.set_minor_locator(np.array([0.001,0.002,0.003,0.005]))
        locator = matplotlib.ticker.FixedLocator([0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])
        ax.xaxis.set_minor_locator(locator)
        ax.set_title("Light Dose Response Curve", fontsize=16)
        plt.show()


    return pulse_arr, mu_arr

def GroundTruthData():

    ### Multisetpoint Dataset

    filename = "/home/smalani/Cybernetic_BioModel/optogenetic_model_fits/June152023Onwards/trainingData/SGy139_TimeCourse_Jun9_23.xlsx"
    df_br1 = pd.read_excel(filename, sheet_name="Bioreactor1")
    df_br2 = pd.read_excel(filename, sheet_name="Bioreactor2")

    # Truncate data to remove initial lag phase
    time_cutoff = 18
    df_br1_trunc = df_br1[df_br1["BatchTime"] > time_cutoff].copy()
    df_br2_trunc = df_br2[df_br2["BatchTime"] > time_cutoff].copy()

    # Smooth data using Savitzky-Golay filter
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
    # Parameters
    window_size = 151
    poly_order = 3
    deriv = 0
    df_br1_trunc["GrowthRateSmooth"] = savgol_filter(df_br1_trunc["GrowthRate"], window_size, poly_order, deriv=deriv)
    df_br2_trunc["GrowthRateSmooth"] = savgol_filter(df_br2_trunc["GrowthRate"], window_size, poly_order, deriv=deriv)

    # Split dataframe by Phase

    df_br1_trunc_grouped = df_br1_trunc.groupby("Phase")
    df_br2_trunc_grouped = df_br2_trunc.groupby("Phase")

    keys = list(df_br1_trunc_grouped.groups.keys())

    mu_mean_br1 = []
    mu_std_br1 = []
    mu_mean_br2 = []
    mu_std_br2 = []
    mu_mean_overall = []
    mu_std_overall = []

    rangeme = 20

    for key in keys:
        df = df_br1_trunc_grouped.get_group(key)
        mu_mean_br1.append(df["GrowthRateSmooth"].iloc[-rangeme:].mean())
        mu_std_br1.append(df["GrowthRateSmooth"].iloc[-rangeme:].std())
        df = df_br2_trunc_grouped.get_group(key)
        mu_mean_br2.append(df["GrowthRateSmooth"].iloc[-rangeme:].mean())
        mu_std_br2.append(df["GrowthRateSmooth"].iloc[-rangeme:].std())

        mu_mean_overall.append((mu_mean_br1[-1] + mu_mean_br2[-1])/2)
        mu_std_overall.append(np.sqrt(mu_std_br1[-1]**2 + mu_std_br2[-1]**2))

#########################################################################################################################3


    #### Activation Deactivation dataset

    filename = "/home/smalani/Cybernetic_BioModel/optogenetic_model_fits/June152023Onwards/trainingData/SGy139_ActDeAct_Mar1_23.xlsx"
    df_br1_actdeact = pd.read_excel(filename, sheet_name="Bioreactor1")
    df_br2_actdeact = pd.read_excel(filename, sheet_name="Bioreactor2")

    df_br1_actdeact.head()

    # Truncate data to remove initial lag phase
    time_cutoff = 18
    df_br1_trunc_actdeact = df_br1_actdeact[df_br1_actdeact["BatchTime"] > time_cutoff].copy()
    df_br2_trunc_actdeact = df_br2_actdeact[df_br2_actdeact["BatchTime"] > time_cutoff].copy()

    # Smooth data using Savitzky-Golay filter
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
    # Parameters
    window_size = 151
    poly_order = 3
    deriv = 0
    df_br1_trunc_actdeact["GrowthRateSmooth"] = savgol_filter(df_br1_trunc_actdeact["GrowthRate"], window_size, poly_order, deriv=deriv)
    df_br2_trunc_actdeact["GrowthRateSmooth"] = savgol_filter(df_br2_trunc_actdeact["GrowthRate"], window_size, poly_order, deriv=deriv)

    # Split dataframe by Phase

    df_br1_trunc_actdeact_grouped = df_br1_trunc_actdeact.groupby("Phase")
    df_br2_trunc_actdeact_grouped = df_br2_trunc_actdeact.groupby("Phase")

    keys_br1_actdeact = list(df_br1_trunc_actdeact_grouped.groups.keys())
    keys_br2_actdeact = list(df_br2_trunc_actdeact_grouped.groups.keys())

    keys_add_br1 = []
    keys_add_br2 = []


    mu_mean_br1_actdeact = []
    mu_std_br1_actdeact = []
    mu_mean_br2_actdeact = []
    mu_std_br2_actdeact = []

    rangeme = 20

    for key in keys_br1_actdeact:
        df = df_br1_trunc_actdeact_grouped.get_group(key)
        mu_mean_br1_actdeact.append(df["GrowthRateSmooth"].iloc[-rangeme:].mean())
        mu_std_br1_actdeact.append(df["GrowthRateSmooth"].iloc[-rangeme:].std())
        # df = df_br2_trunc_actdeact_grouped.get_group(key)
        # mu_mean_br2.append(df["GrowthRateSmooth"].iloc[-rangeme:].mean())
        # mu_std_br2.append(df["GrowthRateSmooth"].iloc[-rangeme:].std())

        if '_0%' in key:
            keys_add_br1.append(0)
        elif '_10%' in key:
            keys_add_br1.append(0.1)

    for key in keys_br2_actdeact:
        df = df_br2_trunc_actdeact_grouped.get_group(key)
        mu_mean_br2_actdeact.append(df["GrowthRateSmooth"].iloc[-rangeme:].mean())
        mu_std_br2_actdeact.append(df["GrowthRateSmooth"].iloc[-rangeme:].std())
        # df = df_br2_trunc_actdeact_grouped.get_group(key)
        # mu_mean_br2.append(df["GrowthRateSmooth"].iloc[-rangeme:].mean())
        # mu_std_br2.append(df["GrowthRateSmooth"].iloc[-rangeme:].std())

        if '_0%' in key:
            keys_add_br2.append(0)
        elif '_10%' in key:
            keys_add_br2.append(0.1)

########################1#############################################################################################################3

    keys_plot = keys + keys_add_br1 + keys_add_br2
    mu_plot = mu_mean_br1 + mu_mean_br1_actdeact + mu_mean_br2_actdeact
    mu_plot = np.array(mu_plot)
    keys_plot = np.array(keys_plot)

    mu_plot = mu_plot[np.argsort(keys_plot)]
    keys_plot = keys_plot[np.argsort(keys_plot)]

    keys_unique = np.unique(keys_plot)
    mu_unique = []
    mu_std_unique = []

    for key in keys_unique:
        mu_unique.append(mu_plot[keys_plot == key].mean())
        mu_std_unique.append(mu_plot[keys_plot == key].std())

    steady_state_output = keys_unique, mu_unique, mu_std_unique

########################1#############################################################################################################3

    # Multisetpoint output

    br1_multisetpoint = df_br1_trunc["BatchTime"].to_numpy(), df_br1_trunc["Phase"].to_numpy(), df_br1_trunc["GrowthRateSmooth"].to_numpy()
    br2_multisetpoint = df_br2_trunc["BatchTime"].to_numpy(), df_br2_trunc["Phase"].to_numpy(), df_br2_trunc["GrowthRateSmooth"].to_numpy()

    multi_setpoint_output = br1_multisetpoint, br2_multisetpoint

########################1#############################################################################################################3

    # Activation Deactivation output

    phase_df = df_br1_trunc_actdeact["Phase"].str.extract('_(.+)%_')[0].astype(float)/100
    br1_actdeact = df_br1_trunc_actdeact["BatchTime"].to_numpy(), phase_df.to_numpy(), df_br1_trunc_actdeact["GrowthRateSmooth"].to_numpy()

    phase_df = df_br2_trunc_actdeact["Phase"].str.extract('_(.+)%_')[0].astype(float)/100
    br2_actdeact = df_br2_trunc_actdeact["BatchTime"].to_numpy(), phase_df.to_numpy(), df_br2_trunc_actdeact["GrowthRateSmooth"].to_numpy()

    act_deact_output = br1_actdeact, br2_actdeact

########################1#############################################################################################################3

    return steady_state_output, multi_setpoint_output, act_deact_output


def integrate(x0, time_arr, pulse_fun, parameters=None):    
    time_out = []
    pulse_out = []
    mu_out = []

    if parameters is None:
        parameters = get_pars()

    for i in range(len(time_arr) - 1):
        t_eval = np.linspace(time_arr[i], time_arr[i+1], 100)
        sol = solve_ivp(SGy139_ODE, [time_arr[i], time_arr[i+1]], x0, t_eval=t_eval, args=(parameters, pulse_fun,))
        x0 = sol.y[:, -1]
        time_out.append(sol.t)
        pulse_out.append(pulse_fun(time_arr[i]) * np.ones(len(sol.t)))
        mu_out.append(sol.y[0, :])

    time_out = np.concatenate(time_out)
    pulse_out = np.concatenate(pulse_out)
    mu_out = np.concatenate(mu_out)

    return time_out, pulse_out, mu_out


def optimizeLight(x0, L0, t_history, L_history, setpoint, time_horizon, sampling_rate, initial_light_arr=None, parameters=None):
    if parameters is None:
        parameters = get_pars()
    time_array = np.arange(sampling_rate, time_horizon, sampling_rate)

    if initial_light_arr is None:
        initial_light_arr = L0 * np.ones(len(time_array))



    def objective(x):
        full_time_array = np.concatenate((t_history-t_history[-1], time_array))
        full_light_array = np.concatenate((L_history, x))

        def pulse_fun(t):
            if t < full_time_array[0]:
                return full_light_array[0]
            elif t > full_time_array[-1]:
                return full_light_array[-1]
            else:
                return full_light_array[full_time_array > t][0] 
        t_arr, _, mu_arr = integrate(x0, np.arange(0, time_horizon, sampling_rate), pulse_fun, parameters)

        if callable(setpoint):
            return np.mean(((mu_arr - setpoint(t_arr + t_history[-1])))**2)
        else:
            return np.mean(((mu_arr - setpoint))**2)
    
    bounds = [(0, 0.1) for _ in range(len(time_array))]

    res = minimize(objective, initial_light_arr, bounds=bounds, method='SLSQP')

    return res.x