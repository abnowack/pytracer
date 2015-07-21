# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 13:57:59 2015

@author: Aaron
"""
from mesh import Mesh
import numpy as np

class Geometry(object):
    def __init__(self):
        self.solids = []
    
    def flatten(self):
        n_points = np.cumsum([0] + [np.size(solid.mesh.points, 0) for solid in self.solids])
        n_lixels = np.cumsum([0] + [np.size(solid.mesh.lixels, 0) for solid in self.solids])
        
        points = [0] + np.zeros((n_points[-1], 2), dtype=np.float32)
        lixels = [0] + np.zeros((n_lixels[-1], 2), dtype=np.int32)
        
        for i, solid in enumerate(self.solids):
            points[n_points[i]:n_points[i + 1]] = solid.mesh.points
            lixels[n_lixels[i]:n_lixels[i + 1]] = solid.mesh.lixels + n_lixels[i]
        
        self.mesh = Mesh(points, lixels)
        self.materials = np.unique(np.concatenate([np.concatenate([solid.inner_material, solid.outer_material]) for solid in self.solids]))
        self.inner_material_index = np.concatenate([[np.where(self.materials == mat)[0][0] for mat in solid.inner_material] for solid in self.solids])
        self.outer_material_index = np.concatenate([[np.where(self.materials == mat)[0][0] for mat in solid.outer_material] for solid in self.solids])


def line_segment_intersect(line_a, line_b):
    p = line_a[0]
    r = line_a[1] - line_a[0]
    q = line_b[0]
    s = line_b[1] - line_b[0]
    
    denom = r[0] * s[1] - r[1] * s[0]
    u_num = (q - p)[0] * r[1] - (q - p)[1] * r[0]
    t_num = (q - p)[0] * s[1] - (q - p)[1] * s[0]
    
    if denom == 0. and u_num == 0.:
        # colinear
        return None # TODO: what should this really return?
    elif denom == 0. and u_num != 0.:
        # parallel
        return None
        
    t = t_num / denom
    u = u_num / denom

    if 0 <= t <= 1. and 0 <= u <= 1.:
        return p + t * r
    else:
        # beyond line segment boundary
        return None

