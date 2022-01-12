import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

def pulse_integrator(tau_off=30, tau_on=5, n=4,
                     pulse_period=100, pulse_on=10,
                     t_start=0, t_step=1, t_end=1000,
                     verbose=False):
    t = 0
    act_out = []
    act_out.append(0)
    t_out = []
    t_out.append(0)
    t_status = 0

    while t < t_end:
        t = t + t_step
        t_phase = t % pulse_period
        if t_phase < pulse_on:
            if t_out[-1] % pulse_period > pulse_on:
                pulse_fun = lambda x: (x ** n) / (x ** n + tau_on ** n) - act_out[-1]
                # t_status = fsolve(pulse_fun, 100)
                sol = root_scalar(pulse_fun, bracket=[0, pulse_period])
                t_status = sol.root
                # assert False
            act = (t_phase + t_status) ** n / ((t_phase + t_status) ** n + tau_on ** n)
        else:
            act = np.exp(-(t_phase - pulse_on) / tau_off)    
            
        t_out.append(t)
        act_out.append(act)

    t_out = np.hstack(t_out)
    act_out = np.hstack(act_out)

    if verbose:
        plt.figure()
        plt.plot(t_out, act_out)

    return t_out, act_out