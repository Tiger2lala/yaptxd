"""
File: utils.py
Author: Minghao Zhang @tiger2lala
Description: Utility functions
"""

import numpy as np
from typing import Tuple

def sinc_pulse(nt: int, tbw: float) -> np.ndarray:
    t = np.linspace(-tbw/2, tbw/2, nt)
    pulse = np.sinc(t)
    # windowing
    n = np.linspace(-0.5,0.5,nt)
    window = 0.53836+0.46164*np.cos(2*np.pi*n)
    pulse = pulse*window
    return pulse


def trapz_gradient(nt_plat: int, plat_amp: float, slew_per_step: float,
                   rev_polarity: bool=False) -> Tuple:
    """
    Generate trapezoidal gradient waveform
    """
    plat = np.ones(nt_plat) * plat_amp
    n_ramp = int(np.ceil(plat_amp / slew_per_step))  # no higher than prescribed slew
    ramp = np.linspace(0, plat_amp, n_ramp)
    grad_out = np.concatenate((ramp[:-1], plat, ramp[-2::-1]))

    t_ramp = n_ramp-1
    return (grad_out * (-1 if rev_polarity else 1), t_ramp)
