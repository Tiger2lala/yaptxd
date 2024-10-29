"""
File: gradient_opt.py
Author: Minghao Zhang @tiger2lala
Description: module for gradient trajectory optimization
"""

import numpy as np
import matplotlib.pyplot as plt
from yaptxd.spokes import SpokesForm
from yaptxd.maps import FieldMap, FieldMapFlattened
from yaptxd.sta import StaOpt
from typing import Union, List, Tuple
from bayes_opt import BayesianOptimization, acquisition


class BOGAT:
    """
    Class to implement Bayesian Optimization for Gradient Trajectory
    doi:10.1002/mrm.30007
    """

    def __init__(self, maps: FieldMap, pulse_form: SpokesForm,
                 target: Union[float, np.ndarray] = 0.5):
        self.maps = maps
        self.pulse_form_in = pulse_form
        self.rf_optim = StaOpt
        self.rf_optim_kwargs = {}
        self._pulse_form_best = []
        self._coeff_best = []

        if not isinstance(target, np.ndarray):
            self.target = target * np.ones_like(self.maps.b0)
        elif not np.alltrue(target.shape == self.maps.b0.shape):
            raise ValueError("target shape does not match b0 shape")
        else:
            self.target = target
    
    
    def set_rf_kwargs(self, **kwargs):
        self.rf_optim_kwargs = kwargs


    def optimize(self, n_iter: int = 30,
                 k_range: float = 0.0015,
                 phase_adoption: bool = False):
        """
        Optimization entry point

        :param n_iter: number of iterations
        :param k_range: range of k-xy plane to explore
        """

        # identify the slices to optimize
        slice_groups = self.maps.slice_groups
        # List[np.ndarray]
        # [igroup][imb, islice]
        n_groups = len(slice_groups)
        mbf = slice_groups[0].shape[0]

        for igroup in range(n_groups):
            # Generate list of targets corresponding to each slice
            slices = slice_groups[igroup]
            imaps = mbf*igroup + np.arange(mbf)
            targets = [self.target[..., islice][self.maps.mask[..., islice]] \
                       for islice in slices]
            # Generate a list of flattened maps
            maps = [self.maps.flattened_maps[i] for i in imaps]
            res = self.optimize_BOGAT(maps, targets, n_iter, k_range,
                                      phase_adoption)
            self._pulse_form_best.append(res[0])
            self._coeff_best.append(res[1])
        
        return self._pulse_form_best, self._coeff_best
        
    @property
    def pulse_form_best(self):
        return self._pulse_form_best
    
    @property
    def coeff_best(self):
        return self._coeff_best

    def optimize_BOGAT(self, maps: List[FieldMapFlattened],
                       targets: List[np.ndarray],
                       n_iter: int = 30,
                       k_range: float = 0.0015,
                       phase_adoption: bool = False) \
                       -> Tuple[SpokesForm, 
                                List[np.ndarray]]:
        bounds = {'x': (-k_range, k_range),
                  'y': (-k_range, k_range)}
        
        existing_k_samples = np.array([0, 0])

        for i_spoke in range(self.pulse_form_in.n_spokes-1):
            bo = BayesianOptimization(f=None,
                                    pbounds=bounds,
                                    acquisition_function=acquisition.ExpectedImprovement(0.5),
                                    allow_duplicate_points=True)
            form = SpokesForm(i_spoke+1, 
                              (self.pulse_form_in.subpulse,
                               self.pulse_form_in.subpulse_tbw),
                               self.pulse_form_in.timestep)
            for _ in range(n_iter):
                points = bo.suggest()
                k_samples = np.vstack((existing_k_samples,
                                      np.array([points['x'], points['y']])))
                form.set_ksamples(k_samples)
                (cost, _, _) = self._objective_evaluate(form, 
                                                maps, targets)
                bo.register(params=points, target=cost)
            
            # record best
            best = bo.max['params']
            existing_k_samples = np.vstack((existing_k_samples,
                                            np.array([best['x'], best['y']])))
            
            # phase adoption
            form.set_ksamples(existing_k_samples)
            (_, coeffs, m_stas) = self._objective_evaluate(form, maps, targets)
            
            if phase_adoption:
                raise NotImplementedError("Phase adoption not working")
            # for i in range(len(targets)): # completely destroys target, why?
            #     targets[i] = np.abs(targets[i]) * np.exp(1j * np.angle(m_stas[i]))

        return form, coeffs
    

    def _objective_evaluate(self,
                            pulse_form: SpokesForm,
                            maps: FieldMapFlattened,
                            targets: np.ndarray):
        """
        Objective function for Bayesian Optimization
        """
        cost = 0.0
        coeffs = []
        m_stas = []
        for iSlice in range(len(maps)):
            rfopt = self.rf_optim(pulse_form,
                                  maps[iSlice],
                                  targets[iSlice])
            rfopt.create_A_matrix()
            rfopt.solve_mls(**self.rf_optim_kwargs)
            cost += rfopt.cost
            print(cost)
            # rfopt.plot_sta()
            coeffs.append(rfopt.coeff)
            m_stas.append(rfopt.m_sta)
        return cost, coeffs, m_stas
