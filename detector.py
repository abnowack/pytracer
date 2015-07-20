# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:03:15 2015

@author: Aaron
"""
import numpy as np
from mesh import angle_matrix

class DetectorPlane(object):
    def __init__(self, center, width, angle=0.):
        self.center = center
        self.width = width
        self.angle = angle
    
    def create_bins(self, nbins=100):
        bins = np.zeros((nbins, 2), dtype=np.float32)
        
        bins[:, 1] = np.linspace(-self.width/2., self.width/2., nbins)
        rot = angle_matrix(self.angle)
        bins = np.dot(bins, rot)
        bins[:, 0] += self.center[0]
        bins[:, 1] += self.center[1]
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