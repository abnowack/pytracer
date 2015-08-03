import numpy as np
import matplotlib.pyplot as plt
from mesh import angle_matrix

class Detector(object):
    """ABC which implements set_nbins to update bin sizes, normal vector calculation. Also drawing method"""
    def set_bins(self, nbins):
        self.bin_edges = self.create_bins(nbins)
        self.bin_centers = (self.bin_edges[1:, :] + self.bin_edges[:-1, :]) / 2.

    def create_bins(self, nbins):
        raise NotImplementedError
    
    def normal(self, bin_i):
        points = self.bin_edges[[bin_i, bin_i + 1]]
        # L[0].y - L[1].y
        px = points[0, 1] - points[1, 1]
        # L[1].x - L[0].x
        py = points[1, 0] - points[0, 0]
        length = np.sqrt(px ** 2. + py ** 2.)
        return np.array([px / length, py / length], dtype=np.float32)

    def solid_angle(self, bin_i, point):
        """Return solid angle in radians."""
        bin_edges = self.bin_edges[[bin_i, bin_i + 1]]
        X = bin_edges[0] - point
        Y = bin_edges[1] - point
        X_Y = X - Y
        cos_angle = (np.dot(X, X) + np.dot(Y, Y) - np.dot(X - Y, X - Y)) / (2 * np.linalg.norm(X) * np.linalg.norm(Y))
        angle = np.arccos(cos_angle)
        if angle > np.pi / 2.:
            angle = np.pi - angle
        return angle

    def solid_angles(self, point):
        bin_solid_angles = [self.solid_angle(i, point) for i in xrange(len(self.bin_centers))]
        return np.array(bin_solid_angles)

    def draw(self, normals=False):
        plt.plot(self.bin_edges[:, 0], self.bin_edges[:, 1])
        if normals:
            for bin_i, bin_center in enumerate(self.bin_centers):
                normal = self.normal(bin_i)
                plt.arrow(bin_center[0], bin_center[1], normal[0], normal[1], width=0.01)

class DetectorPlane(Detector):
    def __init__(self, center, width, angle=0., nbins=None):
        self.center = center
        self.width = width
        self.angle = angle
        if nbins is not None:
            self.set_bins(nbins)
    
    def create_bins(self, nbins=100):
        bins = np.zeros((nbins, 2), dtype=np.float32)
        
        bins[:, 1] = np.linspace(-self.width / 2., self.width / 2., nbins)
        rot = angle_matrix(self.angle)
        bins = np.dot(bins, rot)
        bins[:, 0] += self.center[0]
        bins[:, 1] += self.center[1]
        return bins

class DetectorArc(Detector):
    def __init__(self, center, radius, start_angle, end_angle, nbins=None):
        self.center = center
        self.radius = radius
        self.angles = [start_angle, end_angle]
        if nbins is not None:
            self.create_bins(nbins)

    def create_bins(self, nbins=100):
        angle_bins = np.linspace(self.angles[1], self.angles[0], nbins) * np.pi / 180.
        bins = np.zeros((nbins, 2), dtype=np.float32)
        bins[:, 0] = self.radius * np.cos(angle_bins) + self.center[0]
        bins[:, 1] = self.radius * np.sin(angle_bins) + self.center[1]

        #mean_angle_radian = sum(self.angles) / 2. * np.pi / 180.
        #center_detector = [self.radius * np.cos(mean_angle_radian), 
        #                   self.radius * np.sin(mean_angle_radian)]
        #bins[:, 0] = self.radius * np.cos(angle_bins) + self.center[0] - center_detector[0]
        #bins[:, 1] = self.radius * np.sin(angle_bins) + self.center[1] - center_detector[1]
        return bins