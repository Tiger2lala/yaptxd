"""
File: sta.py
Author: Minghao Zhang @tiger2lala
Description: base class for pulse optimization
"""

import numpy as np
from yaptxd.maps import FieldMapFlattened
from typing import Union

class OptBase:

    def __init__(self, field_maps: FieldMapFlattened,
                 target: Union[float, np.ndarray] = 0.5):
        self.maps = field_maps
        self._coeff = None
        self.cost = 0.
        self.est_fa = None
        if not isinstance(target, np.ndarray):
            self.target = target * np.ones_like(self.maps.b0)
        elif not np.alltrue(target.shape == self.maps.b0.shape):
            raise ValueError("target shape does not match b0 shape")
        else:
            self.target = target

    @property
    def coeff(self):
        return self._coeff
    
    @coeff.setter
    def coeff(self, value):
        self._coeff = value