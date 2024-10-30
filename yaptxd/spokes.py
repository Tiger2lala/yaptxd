"""
File: spokes.py
Author: Minghao Zhang @tiger2lala
Description: Calculate spokes pulses
"""

import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple
from yaptxd.utils import trapz_gradient, GAMMA, SLEW


# TODO: good to have a parent class of Pulse
class SpokesForm:

    def __init__(self, 
                 n_spokes: int,
                 subpulse: Tuple[np.ndarray, float] = None,
                 timestep: float = 10e-6,
                 thickness: float = 5.0e-3):
        self.n_spokes = n_spokes
        self.timestep = timestep
        self.thickness = thickness  # m
        
        self.spoke_location = np.zeros((n_spokes, 2))
        self.rf = None
        self.g  = None
        self.k = None
        self.subpulse_start_time = np.zeros(n_spokes)
        if subpulse is not None:
            self.set_subpulse(*subpulse)
        else:
            self.subpulse = None
            self.subpulse_tbw = 2.0 


    def set_subpulse(self, subpulse: np.ndarray, tbw: float):
        self.subpulse = subpulse
        self.subpulse_len = len(subpulse)
        self.subpulse_tbw = tbw
        self.update_form()

    
    def set_ksamples(self, spoke_location: np.ndarray):
        self.spoke_location = spoke_location
        self.update_form()
    

    def update_form(self):
        if self.subpulse is None:
            return
        
        # Gz
        gzplat = self.subpulse_tbw / \
            (GAMMA * self.thickness * self.timestep*self.subpulse_len)
        (sub_gz, ramp_gz) = trapz_gradient(self.subpulse_len, gzplat,
                               SLEW*self.timestep)
        sub_gz = sub_gz[:-1] # remove the last 0
        
        # Gxy
        k_positions = np.concatenate(([[0,0]], self.spoke_location, [[0,0]]), axis=0)
        k_diff = np.diff(k_positions, axis=0) # in m-1
        g_integral = k_diff / GAMMA / self.timestep # in T/m * step

        # make triangular gradients here
        # tramp * (tramp*slew) = area
        tramp = np.ceil(np.sqrt(np.abs(g_integral) / (SLEW * self.timestep))).astype(int)
        tramp = np.max(tramp)  # use a single largest number for all blips
        tramp = np.max((tramp, ramp_gz))
        ramp_top = g_integral / tramp
        
        # Make gradient
        # Gz is bipolar only for now
        total_steps = (self.subpulse_len-1) * self.n_spokes\
            + (self.n_spokes+1) * (2 * tramp)+1
        
        subpulse_gap = self.subpulse_len - 1 + 2 * tramp

        self.subpulse_start_time = np.arange(self.n_spokes) * subpulse_gap \
            + tramp*2
        
        self.g = np.zeros((total_steps, 3))
        self.rf = np.zeros(total_steps)

        # Do the first blip
        for xy in range(2):
            ramp = np.linspace(0, ramp_top[0, xy], tramp+1)
            self.g[:tramp*2+1, xy] = np.concatenate((ramp, ramp[-2::-1]))

        for i in range(self.n_spokes):
            self.g[self.subpulse_start_time[i]-ramp_gz:self.subpulse_start_time[i]-ramp_gz+len(sub_gz), 2] = \
                sub_gz * (-1)**i
            self.rf[self.subpulse_start_time[i]:self.subpulse_start_time[i]+self.subpulse_len] = \
                self.subpulse

            for xy in range(2):
                ramp = np.linspace(0, ramp_top[i+1, xy], tramp+1)
                self.g[self.subpulse_start_time[i]+self.subpulse_len-1
                       :self.subpulse_start_time[i]+subpulse_gap+1, xy] = \
                    np.concatenate((ramp, ramp[-2::-1]))
        
        self.k_traj()
         
    def plot_pulse(self):
        fig, ax = plt.subplots(3, 1, sharex=True)
        ax[0].plot(self.rf)
        ax[1].plot(self.g[:,2])
        ax[2].plot(self.g[:,0:2])
        plt.show()

    def k_traj(self):
        self.k = -np.cumsum(self.g[::-1,...], axis=0) \
            * GAMMA * self.timestep
        # T/m * Hz/T * s = m-1
        self.k = self.k[::-1,...]
        return self.k
    