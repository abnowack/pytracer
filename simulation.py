from geometry import Geometry
from material import Material
from detector import DetectorArc, DetectorPlane
from source import Source
import math2d
import numpy as np
import matplotlib.pyplot as plt
from itertools import izip

# TODO: Break out fission prob functions and recon methods


class Simulation(object):
    def __init__(self, universe_material):
        """
        Coordinate between Geometry (meshes) and detector plane

        Parameters
        ----------
        universe_material : Material
            Default material, such as vacuum or air
        """
        self.geometry = Geometry(universe_material)
        self.source = None
        self.detector = None
        self.grid = None

    def add_aligned_source_detector(self, diameter=100., nbins=100, width=100., type='plane'):
        self.source = Source(-diameter / 2., 0.)
        if type == 'plane':
            self.detector = DetectorPlane([diameter / 2., 0.], width, nbins)
        elif type == 'arc':
            self.detector = DetectorArc(self.source.pos, diameter, width / 2., -width / 2., nbins)

    def rotate(self, angle):
        if self.detector:
            self.detector.rotate(angle)
        if self.source:
            self.source.rotate(angle)
        if self.grid:
            self.grid.rotate(angle)

    def draw(self, draw_normals=False):
        self.geometry.draw(draw_normals)

        if self.source is not None:
            plt.scatter(self.source.pos[0], self.source.pos[1], color='red', marker='x')

        if self.detector is not None:
            self.detector.draw(draw_normals)

        if self.grid is not None:
            self.grid.draw_lines()

        plt.axis('equal')
        plt.xlabel('X (cm)')
        plt.ylabel('Y (cm)')
    
    def scan(self, angles):
        atten_length = np.zeros((self.detector.nbins, len(angles)))

        self.detector.render()
        
        for i, angle in enumerate(angles):
            self.rotate(angle)
            detector_centers = math2d.center(self.detector.segments)
            for j, detector_center in enumerate(detector_centers):
                atten_length[j, i] = self.geometry.attenuation_length(self.source.pos, detector_center)
        
        return atten_length
    
    def radon_transform(self, angles):
        if type(self.detector) is not DetectorPlane:
            raise TypeError('self.detector is not DetectorPlane')

        radon = np.zeros((np.size(self.detector.nbins, 0), len(angles)))

        for i, angle in enumerate(angles):
            self.rotate(angle)
            detector_points = math2d.center(self.detector.segments)
            source_points = np.dot(detector_points, math2d.angle_matrix(180.))[::-1]
            for j, (source_point, detector_point) in enumerate(izip(detector_points, source_points)):
                radon[j, i] = self.geometry.attenuation_length(source_point, detector_point)
        
        return radon