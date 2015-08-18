import numpy as np
import matplotlib.pyplot as plt
import math2d

class Detector(object):
    """ABC which implements set_nbins to update bin sizes, normal vector calculation. Also drawing method"""

    def calculate_segments(self, nbins):
        raise NotImplementedError

    def draw(self, show_normal=False, color='green'):
        for i, segment in enumerate(self.segments):
            plt.plot(segment[:, 0], segment[:, 1], color=color)
            if show_normal:
                normal = math2d.normal(segment)
                center = math2d.center(segment)
                plt.arrow(center[0], center[1], normal[0], normal[1], width=0.01, color=color)

class DetectorPlane(Detector):
    def __init__(self, center, width, nbins, angle=0):
        self.center = center
        self.width = width
        self.angle = angle

        self.segments = self.calculate_segments(nbins)
    
    def calculate_segments(self, nbins):
        points = np.zeros((nbins+1, 2))
        points[:, 1] = np.linspace(-self.width / 2., self.width / 2., nbins+1)
        rot = math2d.angle_matrix(self.angle)
        points = np.dot(points, rot)
        points[:, 0] += self.center[0]
        points[:, 1] += self.center[1]

        return math2d.create_segments_from_points(points)

class DetectorArc(Detector):
    def __init__(self, center, radius, start_angle, end_angle, nbins):
        self.center = center
        self.radius = radius
        self.start_angle = start_angle
        self.end_angle = end_angle
        
        self.segments = self.calculate_segments(nbins)

    def calculate_segments(self, nbins):
        angles = np.linspace(self.end_angle, self.start_angle, nbins + 1) * np.pi / 180.
        points = np.zeros((nbins+1, 2), dtype=np.float32)
        points[:, 0] = self.radius * np.cos(angles) + self.center[0]
        points[:, 1] = self.radius * np.sin(angles) + self.center[1]

        return math2d.create_segments_from_points(points)