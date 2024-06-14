"""
File: maps.py
Author: Minghao Zhang @tiger2lala
Description: Load and process field maps
"""

import numpy as np
import os
import scipy.io as sio
import matplotlib.pyplot as plt


class FieldMap:
    def __init__(self, map_path: str):
        
        self.b0 = None
        self.b1 = None
        self.x = self.y = self.z = None
        self.xyz_mesh = None
        self.mask = None
        self.coils = 1
        self.dims = np.array([1,1,1])
        
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

        self.dims = np.array([self.x.size, self.y.size, self.z.size])
        self.coils = map_data['coils']
        self.b1 = map_data['S'] \
            .reshape((self.dims[0], self.dims[1], self.coils, self.dims[2]), order='F') \
            .transpose([2,0,1,3]) * 1e-6
        self.xyz_mesh = np.meshgrid(self.x, self.y, self.z, indexing='ij')

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
