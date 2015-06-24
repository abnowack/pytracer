# -*- coding: utf-8 -*-
"""
Created on Fri Jun 05 11:50:01 2015

@author: Aaron

TODO: Implement Geometry Checking
    - Test if any lixels overlap
    - Cannot have hole in a hole, or solid in a solid
TODO: Use Bounded Volume Heirarchy to Reduce Lixel Search
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
    
    def __add__(self, other):
        return Mesh(np.concatenate([self.points, other.points]),
                    np.concatenate([self.lixels, other.lixels + np.size(self.lixels, 0)]))
    
    def continuous_path_order(self):
        """
        mesh points not neccessarily in order, reorganize points such that
        point[i], point[i+1], ... point[0] will trace out continuous path
        """
        new_index = []
    
        next_index = 0
    
        for i, lixel in enumerate(self.lixels):
            if next_index not in new_index:
                new_index.append(next_index)
            else:
                others = np.setdiff1d(self.lixels, self.lixels[new_index])
                next_index = np.where(self.lixels[others[0]] == self.lixels)[0][0]
                new_index.append(next_index)
    
            next_index = np.where(self.lixels[new_index[-1], 1] == self.lixels[:, 0])[0][0]
    
        return self.lixels[new_index], self.points[new_index]

def create_rectangle(w, h):
    points = np.zeros((4, 2), dtype=np.float32)
    lixels = np.zeros((4, 2), dtype=np.int32)
    
    points[1, 1], points[2, 1] = h/2., h/2.
    points[0, 1], points[3, 1] = -h/2., -h/2.
    points[2, 0], points[3, 0] = w/2., w/2.
    points[0, 0], points[1, 0] = -w/2., -w/2.
    
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

def create_hollow(outer_object, inner_object):
    inner_object.points = inner_object.points[::-1]
    return outer_object + inner_object

def angle_matrix(angle, radian=False):
    if not radian:
        angle = angle / 180. * np.pi
    
    return np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])

def translate_rotate_mesh(meshes, translate=np.zeros((2)), rotate=np.identity(2)):
    try:
        iterator = iter(meshes)
    except TypeError:
        meshes.points = np.inner(meshes.points, rotate) + translate
    else:
        for mesh in meshes:
            mesh.points = np.inner(mesh.points, rotate) + translate

class Material(object):
    def __init__(self, attenuation, color='black'):
        self.attenuation = attenuation
        self.color = color
    
    def __eq__(self, other):
        return self.attenuation == other.attenuation

class Solid(object):
    def __init__(self, mesh, inner_material, outer_material):
        self.mesh = mesh
        self.inner_material = np.tile(inner_material, np.size(self.mesh.lixels, 0))
        self.outer_material = np.tile(outer_material, np.size(self.mesh.lixels, 0))
        self.color = inner_material.color

class Geometry(object):
    def __init__(self):
        self.solids = []
    
    def flatten(self):
        n_points = np.cumsum([0] + [np.size(solid.mesh.points, 0) for solid in self.solids])
        n_lixels = np.cumsum([0] + [np.size(solid.mesh.lixels, 0) for solid in self.solids])
        
        points = [0] + np.zeros((sum(n_points), 2), dtype=np.float32)
        lixels = [0] + np.zeros((sum(n_lixels), 2), dtype=np.int32)
        
        self.mesh = Mesh(points, lixels)
        self.materials = np.unique(np.concatenate([np.concatenate([solid.inner_material, solid.outer_material]) for solid in self.solids]))
        self.inner_material_index = np.concatenate([[np.where(self.materials == mat)[0][0] for mat in solid.inner_material] for solid in self.solids])
        self.outer_material_index = np.concatenate([[np.where(self.materials == mat)[0][0] for mat in solid.outer_material] for solid in self.solids])

class DetectorPlane(object):
    def __init__(self, width, angle=0.):
        self.center = center
        self.width = width
        self.angle = angle
    
    def create_bins(self, nbins=100):
        bins = np.zeros((nbins, 2), dtype=np.float32)
        
        bins[:, 1] = np.linspace(-self.width/2., self.width/2., nbins)
        rot = angle_matrix(self.angle)
        bins = np.dot(bins, rot)
        return bins

class DetectorArc(object):
    def __init__(self, center, radius, start_angle, end_angle):
        self.center = center
        self.radius = radius
        self.angles = [start_angle, end_angle]

    def create_bins(self, nbins=100):
        angle_bins = np.linspace(self.angles[0], self.angles[1], nbins) * np.pi / 180.
        bins = np.zeros((nbins, 2), dtype=np.float32)
        mean_angle_radian = sum(self.angles) / 2. * np.pi / 180.
        center_detector = [self.radius * np.cos(mean_angle_radian), 
                           self.radius * np.sin(mean_angle_radian)]
        bins[:, 0] = self.radius * np.cos(angle_bins) + self.center[0] - center_detector[0]
        bins[:, 1] = self.radius * np.sin(angle_bins) + self.center[1] - center_detector[1]
        return bins

class Simulation(object):
    def __init__(self, universe_material, diameter=100., detector_width=100., detector='plane'):
        self.universe_material = universe_material
        self.geometry = Geometry()
        self.source = np.array([-diameter/2., 0.])
        if detector == 'plane':
            self.detector = DetectorPlane([diameter/2., 0.], detector_width)
        elif detector == 'arc':
            self.detector = DetectorArc([diameter/2., 0], diameter, detector_width/2., -detector_width/2.)
    
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
        for solid in self.geometry.solids:
            lixels, points = solid.mesh.continuous_path_order()
            xs = [points[lixels[0, 0]][0]]
            ys = [points[lixels[0, 0]][1]]
            for lixel in lixels:
                if points[lixel[0]][0] == xs[-1] and points[lixel[0]][1] == ys[-1]:
                    xs.append(points[lixel[1]][0])
                    ys.append(points[lixel[1]][1])
                else:
                    xs.extend(points[lixel][:, 0])
                    ys.extend(points[lixel][:, 1])

            plt.fill(xs, ys, color=solid.color)
        
        if self.source is not None:
            plt.scatter(self.source[0], self.source[1], color='red', marker='x')
        
        detector_bins = self.detector.create_bins()
        plt.plot(detector_bins[:, 0], detector_bins[:, 1], color='green')

        plt.axis('equal')
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    air = Material(0.0, 'white')
    u235_metal = Material(0.5, 'green')
    poly = Material(0.1, 'red')
    steel = Material(0.3, 'orange')
    
    box = create_hollow(create_rectangle(20., 10.), create_rectangle(18., 8.))
    hollow_circle = create_hollow(create_circle(3.9), create_circle(2.9))
    translate_rotate_mesh(hollow_circle, [-9+3.9+0.1, 0.])
    small_box_1 = create_rectangle(2., 2.)
    translate_rotate_mesh(small_box_1, [6., 2.])
    small_box_2 = create_rectangle(2., 2.)
    translate_rotate_mesh(small_box_2, [6., -2.])
    
#    translate_rotate_mesh([box, hollow_circle, small_box_1, small_box_2], [20., 10.], angle_matrix(30.))
    
    sim = Simulation(air, 100., 10., 'arc')
    sim.detector.width = 100.
    sim.geometry.solids.append(Solid(box, steel, air))
    sim.geometry.solids.append(Solid(hollow_circle, poly, air))
    sim.geometry.solids.append(Solid(small_box_1, u235_metal, air))
    sim.geometry.solids.append(Solid(small_box_2, u235_metal, air))
    sim.geometry.flatten()
    
    sim.draw()