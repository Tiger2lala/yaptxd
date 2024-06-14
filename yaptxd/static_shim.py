"""
File: static_shim.py
Author: Minghao Zhang @tiger2lala
Description: Calculate static ptx shimming
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from .maps import FieldMap


def static_shim(maps: FieldMap, 
                pulse_form: np.ndarray,
                method: str = 'mls',
                tikhonov: float = 0,
                plotting: bool = False):
    """
    Static shim calculation
    :param maps: FieldMap object
    :param pulse_form: Desired RF pulse form 
    """
    # constants
    gamma = 42.58e6 # Hz/T
    dt = 10e-6 # s

    # Flat target
    target_full = 0.5* np.ones(maps.dims)
    target = target_full[maps.mask]

    # Build A matrix

    # Shim without B0 and G considerations,
    # so A is a number before B1 maps
    a = 2j * np.pi * gamma * dt  # rad/T/step
    a = a * np.sum(pulse_form)  # rad/T/unit

    # Expand to coils
    a_mat = a * maps.b1[:, maps.mask] # ncoil x nvoxel

    # Solve
    if method == 'cls':
        x = shim_cls(a_mat, target)
    elif method == 'mls':
        x = shim_mls(a_mat, target)

    # plotting
    sta_mag = np.zeros_like(maps.mask, dtype=complex)
    sta_mag[maps.mask] = np.matmul(a_mat.T, x)
    plot_sta(target_full, sta_mag)

    return x


def shim_cls(a_mat: np.ndarray, target: np.ndarray,
             tikhonov: float = 0,
             x0: np.ndarray = None) -> np.ndarray:
    """
    CLS solve of shim. Least square.
    :param a_mat: A matrix
    :param target: Target field map
    """
    astack = np.vstack((a_mat.T, tikhonov*np.eye(a_mat.shape[0])))
    bstack = np.append(target, np.zeros(a_mat.shape[0]))

    x = np.linalg.lstsq(astack, bstack, rcond=None)
    return x[0]


def shim_mls(a_mat: np.ndarray, target: np.ndarray,
             tikhonov: float = 0,
             x0: np.ndarray = None,
             niter: int = 30) -> np.ndarray:
    """
    MLS solve of shim with phase adoption
    :param a_mat: A matrix
    :param target: Target field map
    """
    curr_phase = np.zeros_like(target, dtype=complex)
    for i in range(niter):
        astack = np.vstack((a_mat.T, tikhonov*np.eye(a_mat.shape[0])))
        bstack = np.append((target*np.exp(1j*curr_phase)),
                           np.zeros(a_mat.shape[0]))
        x = np.linalg.lstsq(astack, bstack, rcond=None)
        curr_phase = np.angle(np.matmul(a_mat.T, x[0]))

    return x[0]


def mls_cost(b_vec: np.ndarray, a_mat: np.ndarray, 
             target: np.ndarray, 
             tikhonov: float) -> float:
    """
    Arguments:
        b_vec: np.ndarray, (DOF*2,)
        a_mat: np.ndarray, (DOF, nVoxels)
        target: np.ndarray, (nVoxels,)
        tikhonov: float
    """
    complex_b = b_vec[::2] + 1j*b_vec[1::2]
    sta_mag = np.matmul(a_mat.T, complex_b)
    cost = np.linalg.norm(np.abs(sta_mag) - np.abs(target)) **2 + \
        (tikhonov * np.linalg.norm(b_vec)) **2
    return cost


def plot_sta(target, sta_mag):
    """
    """
    fig, ax = plt.subplots(1,3)
    im1 = ax[0].imshow(target, cmap='hot')
    im1.set_clim(0, 1)
    plt.colorbar(im1, ax=ax[0])
    ax[0].set_title('Target magnetization')

    im2 = ax[1].imshow(np.abs(sta_mag), cmap='hot')
    im2.set_clim(0, 1)
    plt.colorbar(im2, ax=ax[1])
    ax[1].set_title('STA magnetization')

    im3 = ax[2].imshow(np.angle(sta_mag), cmap='hsv', vmin=-np.pi, vmax=np.pi)
    plt.colorbar(im3, ax=ax[2])
    ax[2].set_title('STA phase')
    plt.show()
    return