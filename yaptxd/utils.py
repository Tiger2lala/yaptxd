"""
File: utils.py
Author: Minghao Zhang @tiger2lala
Description: Utility functions
"""

import numpy as np

def sinc_pulse(nt: int, tbw: float) -> np.ndarray:
    t = np.linspace(-tbw/2, tbw/2, nt)
    pulse = np.sinc(t)
    # windowing
    n = np.linspace(-0.5,0.5,nt)
    window = 0.53836+0.46164*np.cos(2*np.pi*n)
    pulse = pulse*window
    return pulse