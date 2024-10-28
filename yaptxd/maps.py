"""
File: maps.py
Author: Minghao Zhang @tiger2lala
Description: Load and process field maps
"""

import numpy as np
import os
import scipy.io as sio
import matplotlib.pyplot as plt
from typing import List, Union


class FieldMap:
    def __init__(self, map_path: str = None,):
        
        self.b0 = None
        self.b1 = None
        self.x = self.y = self.z = None
        self.xyz_mesh = None
        self.mask = None
        self.coils = 1
        self.dims = np.array([1,1,1])
        self.slice_groups = []
        self.flattened_maps = None

        
        if map_path is not None:
            _, ext =  os.path.splitext(map_path)
            if ext != '.mat':
                raise ValueError('Only .mat format supported')
            self.load_map(map_path)

    def load_map(self, map_path: str):
        """
        Load field map from .mat file
        """
        map_data = sio.loadmat(map_path,simplify_cells=True)['Adj']
        self.b0 = map_data['B0']
        self.x = map_data['values_n'] * 1e-3
        self.y = map_data['values_m']* 1e-3
        self.z = np.array([map_data['values_s']]) \
            if np.isscalar(map_data['values_s'])  \
                else map_data['values_s'] * 1e-3
        self.mask = map_data['W'].astype(bool)

        if self.mask.ndim == 2:
            self.mask = self.mask[..., np.newaxis]
        if self.b0.ndim == 2:
            self.b0 = self.b0[..., np.newaxis]

        self.calc_dim_mesh()
        self.coils = map_data['coils']
        self.b1 = map_data['S'] \
            .reshape((self.dims[0], self.dims[1], self.coils, self.dims[2]), order='F') \
            .transpose([2,0,1,3]) * 1e-6

        return map_data

    def quick_plot(self):
        """
        Plot B0 map, mask, and coil sensitivity of channel 1
        """
        fig, axs = plt.subplots(2, 2)
        im1 = axs[0,0].imshow(self.b0, cmap='RdYlGn')
        axs[0,0].set_title('B0 map')
        plt.colorbar(im1, ax=axs[0,0])
        axs[0,1].imshow(self.mask, cmap='gray')
        axs[0,1].set_title('Mask')
        im2 = axs[1,0].imshow(np.abs(self.b1[0,:,:,0]), cmap='hot')
        axs[1,0].set_title('B1 map (coil 0)')
        plt.colorbar(im2, ax=axs[1,0])
        im3 = axs[1,1].imshow(np.angle(self.b1[0,:,:,0]), cmap='hsv', vmin=-np.pi, vmax=np.pi)
        axs[1,1].set_title('B1 phase (coil 0)')
        plt.colorbar(im3, ax=axs[1,1])
        plt.show()
        pass


    def group_slices(self, mbf: int = 1, n_groups: int = 1):
        """
        Makes slice groups for pulse optimization.
        The groups contain slices from regular intervals 
        determined by the multiband factor.
        Then the slices are combined into n_groups per band.

        :param mbf: multiband factor. Must divide total slices.
        :param n_groups: number of groups
        """
        if self.z.size % mbf != 0:
            raise ValueError('Multiband factor must divide total slices')
        
        nslice_per_band = self.z.size // mbf
        groups_in_band = np.array_split(np.arange(nslice_per_band), n_groups)
        self.slice_groups = []
        self.flattened_maps = []
        # 1st index is group, then in array: 1st is band, 2nd is slice
        for i in range(n_groups):
            this_group = []
            for j in range(mbf):
                this_group.append(groups_in_band[i] + j * nslice_per_band)
                self.flattened_maps.append(FieldMapFlattened(self, this_group[-1]))
            self.slice_groups.append(np.array(this_group))
        

    def calc_dim_mesh(self):
        """
        Calculate the meshgrid of x, y, z
        """
        self.xyz_mesh = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        self.dims = np.array([self.x.size, self.y.size, self.z.size])

    
    def flattened(self):
        return FieldMapFlattened(self)



class FieldMapFlattened:

    def __init__(self, from_map: FieldMap = None,
                 slices: np.ndarray = None):
        self.b0 = None
        self.b1 = None
        self.xyz_mesh = None
        self.mask = None
        self.coils = 1
        self.subsample_factor = 1.0

        if from_map is not None:
            self.import_slices(from_map, slices)


    def import_slices(self, from_map: FieldMap, 
                      slices: np.ndarray = None,
                      use_z: bool = True):
        """
        Import slices from another map, combine slices and flatten.
        """
        if slices is None:
            slices = np.arange(from_map.z.size)
        # import
        self.b0 = from_map.b0[:,:,slices]
        self.b1 = from_map.b1[:,:,:,slices]
        self.mask = from_map.mask[:,:,slices]
        self.coils = from_map.coils

        z = from_map.z[slices] if use_z else np.zeros_like(slices)

        self.xyz_mesh = np.meshgrid(from_map.x, from_map.y, z, indexing='ij')

        # flatten
        self.b0 = self.b0[self.mask]
        self.b1 = self.b1[:, self.mask]
        self.xyz_mesh = [x[self.mask] for x in self.xyz_mesh]


    def subsample(self, factor: float = 1.0, method: str = 'uniform'):
        """
        Subsample the field map
        """
        if factor < 1:
            raise ValueError("Subsampling factor should be greater or equal to 1")

        if method == 'uniform':
            chosen = np.linspace(0, self.b0.size-1, int(self.b0.size/factor)).astype(int)
        elif method == 'random':
            chosen = np.random.choice(np.arange(self.b0.size), 
                                   size=int(self.b0.size/factor), 
                                   replace=False)
            
        self.b0 = self.b0[chosen]
        self.b1 = self.b1[:,chosen]
        self.xyz_mesh = [x[chosen] for x in self.xyz_mesh]

        new_mask = np.zeros_like(self.mask, dtype=bool)
        old_idx = np.nonzero(self.mask)
        new_idx = tuple([x[chosen] for x in old_idx])
        new_mask[new_idx] = True
        self.mask = new_mask

        self.subsample_factor = factor
