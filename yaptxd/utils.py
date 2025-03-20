"""
File: utils.py
Author: Minghao Zhang @tiger2lala
Description: Utility functions
"""

import numpy as np
from typing import Tuple
from bloch.bloch import bloch
# from multiprocessing.pool import ThreadPool

GAMMA = 42.58e6  # Hz/T
SLEW = 200 # T/m/s

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



def gradient_delay_adjust(coeff: np.ndarray, z: float, gz: float,
                          delay: np.ndarray, eddy_current: np.ndarray,
                          n_coils: int = 8) -> np.ndarray:
    """
    Adjust phases for pulses to account for gradient delays.
    Only works in the slice direction for now.

    :param coeff: Coefficients of 1 spokes pulse.
    :param z: Slice position in m.
    :param gz: Slice selection gradient in T/m.
    :param delay: Gradient delays [R, P, S] in us.
    :param eddy_current: Eddy current phase [R, P, S] in degrees.
    """

    if np.any(delay[0:2]):
        raise NotImplementedError("Only slice direction is supported.")
    
    ccoeff = coeff.reshape(-1, n_coils).copy() # (nPulse, nCoils)

    phase_add = np.ones(ccoeff.shape[0], dtype=np.complex128)
    phase_even = eddy_current[2] * np.pi / 180\
        + 2 * gz * z * delay[2] * 1e-6 * GAMMA * 2 * np.pi

    phase_add[1::2] = phase_add[1::2] * np.exp(1j * phase_even)
    ccoeff *= phase_add[..., np.newaxis]

    return ccoeff.reshape(-1)


def bloch_eval(b1: np.ndarray, g: np.ndarray, dt: float, df: np.ndarray,
               pos: np.ndarray, t1: float=100, t2: float=100, outtype='flip') -> np.ndarray:
    """
    Bloch simulation for a single voxel.
    :param b1: B1 field in T.
    :param g: Gradient in T/m.
    :param dt: Time step in s.
    :param df: Off-resonance in Hz.
    :param t1: T1 in s.
    :param t2: T2 in s.
    :param m0: Initial magnetization.
    :param pos: Position in m.
    :param outtype: 'flip' or 'mag'.
    :return: Final magnetization (mxy, mz).
    """
    mout = np.zeros((b1.shape[1], 3))

    # def _inner_bloch(i):
    #     (mx, my, mz) = bloch(b1[:,i]*1e4, g.T*0.1, dt, t1, t2, df[i], 
    #                 pos[:,[i]]*100, 0)
    #     return [mx[0], my[0], mz[0]]

    for i in range(len(mout)):
        (mx, my, mz) = bloch(b1[:,i]*1e4, g.T*100, dt, t1, t2, df[i], 
                    pos[:,[i]]*100, 0)
        mout[i] = [mx[0], my[0], mz[0]]
    # with ThreadPool() as p:
    #     mout = np.array(p.map(_inner_bloch, range(b1.shape[1])))
    
    if outtype == 'flip':
        mout = np.arctan2(np.linalg.norm(mout[:,0:2], axis=1), mout[:,2]) * np.exp(1j*np.angle(mout[:,0]+1j*mout[:,1]))
    elif outtype == 'mag':
        pass
    else:
        raise ValueError('Invalid output type.')
    return mout 