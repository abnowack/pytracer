# -*- coding: utf-8 -*-
"""
Created on Fri Jun 05 11:50:01 2015

@author: Aaron

TODO: Implement Geometry Checking
    - Test if any lixels overlap
    - Cannot have hole in a hole, or solid in a solid
TODO: Use Bounded Volume Heirarchy to Reduce Lixel Search
TODO: Material Renderer
"""
import numpy as np

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

class Mesh(object):
    def __init__(self, points, lixels):
        self.points = points
        self.lixels = lixels

def create_rectangle(w, h):
    points = np.zeros((4, 2), dtype=np.float32)
    lixels = np.zeros((4, 2), dtype=np.int32)
    
    points[1, 1], points[3, 1] = h, h
    points[2, 0], points[3, 0] = w, w
    
    lixels[:, 0] = np.arange(np.size(points, 0))
    lixels[:, 1] = np.roll(lixels[:, 0], 1)
    
    return Mesh(points, lixels)

def create_circle(radius, n_segments=20):
    points = np.zeros((n_segments, 2), dtype=np.float32)
    lixels = np.zeros((n_segments, 2), dtype=np.int32)
    
    radians = np.linspace(0., 2 * np.pi, n_segments+1)[:-1]
    points[:, 0] = np.cos(radians) * radius
    points[:, 1] = np.sin(radians) * radius
    
    lixels[:, 0] = np.arange(np.size(points, 0))
    lixels[:, 1] = np.roll(lixels[:, 0], 1)
    
    return Mesh(points, lixels)

class Material(object):
    def __init__(self, attenuation):
        self.attenuation = attenuation
    
    def __eq__(self, other):
        return self.attenuation == other.attenuation

class Solid(object):
    def __init__(self, mesh, inner_material, outer_material, color='black'):
        self.mesh = mesh
        self.inner_material = np.tile(inner_material, np.size(self.mesh.lixels, 0))
        self.outer_material = np.tile(outer_material, np.size(self.mesh.lixels, 0))
        self.color = color

class Geometry(object):
    def __init__(self):
        self.solids = []
        self.rotations = []
        self.translations = []
    
    def add_solid(self, solid, rotation=np.identity(2), translation=np.zeros((2))):
        self.solids.append(solid)
        self.rotations.append(rotation)
        self.translations.append(translation)
    
    def flatten(self):
        n_points = np.cumsum([0] + [np.size(solid.mesh.points, 0) for solid in self.solids])
        n_lixels = np.cumsum([0] + [np.size(solid.mesh.lixels, 0) for solid in self.solids])
        
        points = [0] + np.zeros((sum(n_points), 2), dtype=np.float32)
        lixels = [0] + np.zeros((sum(n_lixels), 2), dtype=np.int32)
        
        for i, (solid, rot, trans) in enumerate(zip(self.solids, self.rotations, self.translations)):
            points[n_points[i]:n_points[i+1]] = np.inner(solid.mesh.points, rot) + trans
            lixels[n_lixels[i]:n_lixels[i+1]] = solid.mesh.lixels + n_lixels[i]
        
        self.mesh = Mesh(points, lixels)
        self.materials = np.unique(np.concatenate([np.concatenate([solid.inner_material, solid.outer_material]) for solid in self.solids]))
        self.inner_material_index = np.concatenate([[np.where(self.materials == mat)[0][0] for mat in solid.inner_material] for solid in self.solids])
        self.outer_material_index = np.concatenate([[np.where(self.materials == mat)[0][0] for mat in solid.outer_material] for solid in self.solids])

class Simulation(object):
    def __init__(self):
        self.geometry = Geometry()
        self.detector = None
        self.source = None
    
    def create_source_detector(self, angle):
        source = np.array([-self.diameter, 0.], dtype=np.float32)
    
    def attenuation_length(self, start, end):
        if not hasattr(self.geometry, 'mesh'):
            self.geometry.flatten()
        intersecting_lixels = []
        distances = []
        for i, lixel in enumerate(self.geometry.mesh.lixels):
            intersect = line_segment_intersect(self.geometry.mesh.points(lixel), np.array(start, end))
            if intersect:
                intersecting_lixels.append(i)
                distance = np.norm(intersect - start)
                distances.append(intersect)
        sorted_lixels = intersecting_lixels[distances.argsort()]
        
        print sorted_lixels
        
    def draw(self):
        # TODO: plot lixels (lines)
        plt.scatter(self.geometry.mesh.points[:, 0], self.geometry.mesh.points[:, 1])
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    air = Material(0.0)
    u235_metal = Material(0.5)
    poly = Material(0.1)
    steel = Material(0.3)
    
    box = create_rectangle(10., 10.)
    circle = create_circle(10.)
    
    sim = Simulation()
    sim.geometry.add_solid(Solid(box, steel, air))
    sim.geometry.add_solid(Solid(circle, poly, air))
    sim.geometry.flatten()
    
    sim.draw()