"""
File: io.py
Author: Minghao Zhang @tz2lala
Description: Load and save data
"""

import numpy as np
from yaptxd.spokes import SpokesForm
from yaptxd.utils import GAMMA


def write_spokes_xy_moment(filename: str, spokes: SpokesForm):
    """
    Write the x and y moment of the spokes form to a file
    """
    k_positions = np.concatenate(([[0,0]], spokes.spoke_location, [[0,0]]), axis=0)
    k_diff = np.diff(k_positions, axis=0) # in m-1
    g_integral = k_diff / GAMMA  # in T/m*s
    g_integral *= 1e9; # in uT/m*ms

    np.savetxt(filename, g_integral, fmt='%.6f', delimiter=',')


def write_spokes_coeff(filename: str, coeff: np.ndarray, n_coils: int = 8):
    """
    Write spokes coefficients to a file
    Compatible with mz407's SBB format
    Real, Imag
    c1s1, c1s1
    c1s2, c1s2
    ...
    c2s1, c2s1
    ...
    """
    coeff = coeff.reshape(-1, n_coils).T.reshape(-1)
    separate = np.array([np.real(coeff), np.imag(coeff)]).T
    np.savetxt(filename, separate, fmt='%.6f', delimiter=',')