import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import root
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

def pulse_integrator(tau_off=30, tau_on=5, n=4,
                     pulse_period=100, pulse_on=10,
                     t_start=0, t_step=1, t_end=1000, act_init = 0,
                     verbose=False):

    if pulse_on > pulse_period:
        raise ValueError('pulse_period cannot be less than pulse_on')

    t = 0
    t_end = t_end - t_start

    act_out = []
    act_out.append(act_init)
    t_out = []
    t_out.append(t)
    light_out = []
    light_time_out = []

    if pulse_period == 0:
        light_out.append(0)
    else:
        light_out.append(1)

    light_time_out.append(0)

    act_status = act_init
    pulse_fun = lambda x: (x ** n) / (x ** n + tau_on ** n) - act_init
    sol = root_scalar(pulse_fun, bracket=[0, 100000])
    t_status = sol.root

    # sol  = root_scalar(pulse_fun, 10.)
                                #    full_output=True, 
                                #    maxfev=1000)


    # t_status = np.abs(sol.x) 

    # if sol.success is False:
    #     print('pulse_integrator root failed')
    #     print(sol.x)
    #     print(t_status)
    #     print(sol.success)
    #     print(sol.message)

    # if pulse_on > 0 and pulse_on < pulse_period:
    #     t_status = t_status % pulse_period
    
    t_curr = 0

    while t < t_end:
        if t + t_step > t_end:
            t_step = t_end - t

        t_phase = t % pulse_period

        # if pulse_period == pulse_on:
        #     light_time_out.append(t)
        #     light_out.append(1)
        #     act = (t + t_status) ** n / ((t + t_status) ** n + tau_on ** n)
        #     t = t + t_step
        #     t_out.append(t)
        #     act_out.append(act)
        #     continue

        # if pulse_on == 0:
        #     light_time_out.append(t)
        #     light_out.append(0)
        #     act = act_status * np.exp(-((t + t_step)) / tau_off)
        #     t = t + t_step
        #     t_out.append(t)
        #     act_out.append(act)
        #     continue


        if t_phase < pulse_on:
            if light_out[-1] == 0:
                pulse_fun = lambda x: (x ** n) / (x ** n + tau_on ** n) - act_out[-1]
                sol = root_scalar(pulse_fun, bracket=[0, 100000])
                t_status = sol.root

                t_curr = t
                # sol = root_scalar(pulse_fun, bracket=[0, 10000])
                # t_status = sol.root
                # print('my t status is: ' + str(t_status))

                light_time_out.append(t)
                light_out.append(0)
            light_time_out.append(t)
            light_out.append(1)
            light_time_out.append(t+t_step)
            light_out.append(1)

                # assert False
            act = ((t-t_curr+t_step) + t_status) ** n / (((t-t_curr+t_step) + t_status) ** n + tau_on ** n)
        else:
            if light_out[-1] == 1:
                act_status = act_out[-1]

                t_curr = t
                # print('my act status is: ' + str(act_status))

                light_time_out.append(t)
                light_out.append(1)
            light_time_out.append(t)
            light_out.append(0)
            light_time_out.append(t+t_step)
            light_out.append(0)
            act = act_status * np.exp(-((t-t_curr)) / tau_off)
            # print('light is off!')
        
        t = t + t_step
        t_out.append(t)
        act_out.append(act)

        # t = t + t_step

    # light_out.insert(0,light_out[0])
    # light_time_out.insert(0,t_start)

    t_out = np.hstack(t_out) + t_start
    act_out = np.hstack(act_out)
    light_out = np.hstack(light_out)
    light_time_out = np.hstack(light_time_out) + t_start
    zero_line = np.zeros(light_time_out.shape)

    if verbose:
        plt.figure(figsize=(10,8))
        plt.plot(t_out, act_out, linewidth=3)
        plt.fill_between(light_time_out, light_out, zero_line, color='cyan', alpha=0.2)
        plt.ylabel('Transcription', fontsize=25)
        plt.xlabel('Time', fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

    return t_out, act_out, light_out, light_time_out