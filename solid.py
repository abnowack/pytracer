# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:04:56 2015

@author: Aaron
"""
import numpy as np

class Solid(object):
    def __init__(self, mesh, inner_material, outer_material):
        self.mesh = mesh
        self.inner_material = np.tile(inner_material, np.size(self.mesh.lixels, 0))
        self.outer_material = np.tile(outer_material, np.size(self.mesh.lixels, 0))
        self.color = inner_material.color
    