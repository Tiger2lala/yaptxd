"""
File: sta.py
Author: Minghao Zhang @tiger2lala
Description: class for large tip calculations
"""

import numpy as np
from scipy.optimize import minimize
from yaptxd.spokes import SpokesForm
from yaptxd.maps import FieldMapFlattened
from yaptxd.opt_base import OptBase
from yaptxd.utils import GAMMA, bloch_eval
from typing import Union

class LtaOpt(OptBase):
    """
    Class for large tip spokes pulse optimization
    """

    def __init__(self, pulse_form: SpokesForm, 
                 field_maps: FieldMapFlattened,
                 target: Union[float, np.ndarray] = 0.5):
        
        super().__init__(field_maps, target)
        self.pulse_form = pulse_form
        self.mls = 1
        self.tikhonov = 0.

    def optimize(self):
        """
        Optimization routine
        Can add switches here later.
        """
        self.optimize_bloch_ipopt()

    def optimize_bloch_ipopt(self):
        """
        Optimization routine using bloch simulation and ipopt
        """
        # optimizer usually only takes real values
        coeff = np.concatenate([np.real(self.coeff), np.imag(self.coeff)])

        # optimization
        res = minimize(self._bloch_cost, coeff, method='trust-constr',
                       options={'disp': True, 'maxiter': 10}, callback=self._opt_callback)
        self.coeff = res.x[:coeff.size//2] + 1j * res.x[coeff.size//2:]
        self.cost = res.fun
        
    
    def _bloch_cost(self, coeff: np.ndarray):
        """
        Cost function for bloch simulation
        """
        # return to complex
        coeff = coeff[:coeff.size//2] + 1j * coeff[coeff.size//2:]
        rcoeff = coeff.reshape(-1, self.maps.coils) # (nPulse, nCoil)
        rf = self.pulse_form.gen_rf(rcoeff) # (nT, ncoil)
        b1_to_sim = rf @ self.maps.b1 # (nT, nVoxel)
        faout = bloch_eval(b1_to_sim, self.pulse_form.g, self.pulse_form.timestep, 
                           self.maps.b0, self.maps.xyz_mesh)
        # faout is (nVoxel) of complex flip angle

        # regularised magnitude cost
        return np.linalg.norm(np.abs(faout) - np.abs(self.target))**2 + \
            self.tikhonov * np.linalg.norm(coeff)**2
    
    @property
    def coeff(self):
        return super().coeff
    
    @coeff.setter
    def coeff(self, value):
        self._coeff = value

        rcoeff = self.coeff.reshape(-1, self.maps.coils)
        rf = self.pulse_form.gen_rf(rcoeff)
        b1_to_sim = rf @ self.maps.b1
        self.est_fa = bloch_eval(b1_to_sim, self.pulse_form.g, self.pulse_form.timestep, 
                                 self.maps.b0, self.maps.xyz_mesh)
    
    @staticmethod
    def _opt_callback(intermediate_result):
        print(f"Cost: {intermediate_result.fun}")