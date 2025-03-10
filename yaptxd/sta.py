"""
File: sta.py
Author: Minghao Zhang @tiger2lala
Description: class for small tip calculations
"""

import numpy as np
import matplotlib.pyplot as plt
from yaptxd.spokes import SpokesForm
from yaptxd.maps import FieldMapFlattened
from yaptxd.utils import GAMMA
from typing import Union


class StaOpt:
    """
    Class for small tip approximation pulse optimization
    """

    def __init__(self, pulse_form: SpokesForm, 
                 field_maps: FieldMapFlattened,
                 target: Union[float, np.ndarray] = 0.5):
        self.pulse_form = pulse_form
        self.maps = field_maps
        if not isinstance(target, np.ndarray):
            self.target = target * np.ones_like(self.maps.b0)
        elif not np.alltrue(target.shape == self.maps.b0.shape):
            raise ValueError("target shape does not match b0 shape")
        else:
            self.target = target
        self.a_mat = None
        self._coeff = None # flattened from (nPulse, nCoils)
        self.cost = 0.
        self.m_sta = None

    def create_A_matrix(self):
        """
        Create the A matrix for the pulse design
        """
        dt = self.pulse_form.timestep

        # Hz, (nT, nVoxels)
        b0_phase_contribution = - self.maps.b0[np.newaxis, ...] * \
            dt * \
            np.linspace(self.pulse_form.k.shape[0], 0,
                        self.pulse_form.k.shape[0])[..., np.newaxis]  

        # (nT, 3) * (3, nVoxels) = (nT, nVoxels)
        # m^-1 * m  
        g_phase_contribution = \
            np.matmul(self.pulse_form.k, 
                      np.concatenate((self.maps.xyz_mesh[0][np.newaxis], 
                                      self.maps.xyz_mesh[1][np.newaxis],
                                      self.maps.xyz_mesh[2][np.newaxis]),
                                      axis=0)) 
        
        # rad / T (nT, nVoxels)
        a_ij = 2j * np.pi * GAMMA * dt * \
            np.exp(2j * np.pi * (b0_phase_contribution + g_phase_contribution))  

        # now sum up each subpulse
        pulse_start_idx = self.pulse_form.subpulse_start_time
        a_before_coil = np.zeros((pulse_start_idx.size, 
                                  np.count_nonzero(self.maps.mask)), dtype=complex)
        for i in range(pulse_start_idx.size):
            # (iPulse, nVoxels)
            a_before_coil[i, :] = np.matmul(self.pulse_form.subpulse[np.newaxis],
                                            a_ij[pulse_start_idx[i]:pulse_start_idx[i]+self.pulse_form.subpulse_len, ...])

        # expand to coils
        # (1, nCoils, nVoxels) * (nPulse, 1, nVoxels)
        # (nPulse, nCoils, nVoxels)
        # rad/V 
        a_mat = self.maps.b1[np.newaxis] * \
            a_before_coil[:, np.newaxis, :]
        # (DOF, nVoxels)
        a_mat = a_mat.reshape(-1, np.count_nonzero(self.maps.mask))  

        self.a_mat = a_mat


    def solve_mls(self, tikhonov: float = 0,
                  niter: int = 30) -> np.ndarray:
        """
        MLS solve of spokes with phase adoption
        :param tikhonov: Tikhonov regularization
        :param niter: number of iterations
        """
        if self.a_mat is None:
            self.create_A_matrix()
        curr_phase = np.zeros_like(self.target, dtype=complex)
        for i in range(niter):
            astack = np.vstack((self.a_mat.T, tikhonov*np.eye(self.a_mat.shape[0])))
            bstack = np.append((np.abs(self.target)*np.exp(1j*curr_phase)),
                                np.zeros(self.a_mat.shape[0]))
            x = np.linalg.lstsq(astack, bstack, rcond=None)
            curr_phase = np.angle(np.matmul(self.a_mat.T, x[0]))
        
        self.coeff = x[0]
        self.cost = self.mls_cost(self.a_mat, x[0], self.target, tikhonov)
        self.m_sta = np.matmul(self.a_mat.T, self.coeff)
        return self.coeff
    
    @staticmethod
    def mls_cost(a_mat: np.ndarray, b_vec: np.ndarray,
             target: np.ndarray, 
             tikhonov: float) -> float:
        """
        Arguments:
            b_vec: np.ndarray, (DOF,)
            a_mat: np.ndarray, (DOF, nVoxels)
            target: np.ndarray, (nVoxels,)
            tikhonov: float
        """
        sta_mag = np.matmul(a_mat.T, b_vec)
        cost = np.linalg.norm(np.abs(sta_mag) - np.abs(target)) **2 + \
            (tikhonov * np.linalg.norm(b_vec)) **2
        return cost


    def plot_sta(self, slice: int = -1):
        """
        Plotting small tip angle magnetization
        """
        sta_mag = np.zeros_like(self.maps.mask, dtype=complex)
        sta_mag[self.maps.mask] = self.m_sta

        target = np.zeros_like(self.maps.mask, dtype=complex)
        target[self.maps.mask] = self.target

        if slice == -1:
            slice = self.maps.mask.shape[2] // 2

        fig, ax = plt.subplots(1,3)
        im1 = ax[0].imshow(np.abs(target)[:,:,slice], cmap='hot')
        im1.set_clim(0, 1)
        plt.colorbar(im1, ax=ax[0])
        ax[0].set_title('Target magnetization')

        im2 = ax[1].imshow(np.abs(sta_mag)[:,:,slice], cmap='hot')
        im2.set_clim(0, 1)
        plt.colorbar(im2, ax=ax[1])
        ax[1].set_title('STA magnetization')

        im3 = ax[2].imshow(np.angle(sta_mag)[:,:,slice], cmap='hsv', vmin=-np.pi, vmax=np.pi)
        plt.colorbar(im3, ax=ax[2])
        ax[2].set_title('STA phase')
        plt.show()
        return
    
    @property
    def coeff(self):
        return self._coeff
    @coeff.setter
    def coeff(self, value):
        self._coeff = value
        if self.a_mat is not None:
            self.m_sta = np.matmul(self.a_mat.T, self.coeff)
