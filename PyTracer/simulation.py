from itertools import izip

import matplotlib.pyplot as plt
import numpy as np

from PyTracer import math2d
from PyTracer.detector import DetectorArc, DetectorPlane
from PyTracer.geometry import Geometry
from source import Source


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

    def propagate_fission_ray(self, start, end, n):
        segments, macro_fissions = self.geometry.fission_segments(start, end)
        segment_probs = []
        for i in xrange(len(segments)):
            single_fission_prob = self.propagate_fissions_segment(segments[i], n)
            segment_probs.append(single_fission_prob)
        total_fission_prob = np.sum(segment_probs, axis=0)
        return total_fission_prob

    def propagate_fissions_segment(self, segment, n=5):
        point_0, point_1 = segment[0], segment[1]
        # generate points along fission segment
        # use trapezoid rule on uniform spacing
        # int [f(x = [a, b]) dx]  ~= (b - a) / (2 * N) [ f(a) + f(b) +  ]
        points = [point_0 + (point_1 - point_0) * t for t in np.linspace(0.01, 0.99, n)] # TODO : error if t = 1
        values = np.zeros((len(points), len(self.detector.segments)))
        integral = np.zeros((len(self.detector.segments)))
        for i in xrange(len(points)):
            values[i, :] = self.propagate_fissions_point_detector(points[i])
        integral[:] = np.linalg.norm(point_1 - point_0) / (n - 1) * (values[0, :] + 2. * np.sum(values[1:-1, :], axis=0) + values[-1, :])
        return integral

    def propagate_fissions_point_detector(self, point):
        """
        Calculate probability of induced fission being detected over detector plane.

        nu = 1 for now, not using macro_fission
        """
        detector_solid_angle = math2d.solid_angle(self.detector.segments, point) / (2. * np.pi) # returns 200,200
        in_attenuation_length = self.geometry.attenuation_length(self.source.pos, point)
        segment_centers = math2d.center(self.detector.segments)
        out_attenuation_lengths = np.array([self.geometry.attenuation_length(point, center) for center in segment_centers])

        prob = np.exp(-in_attenuation_length) * np.multiply(detector_solid_angle, np.exp(-out_attenuation_lengths))

        return prob
