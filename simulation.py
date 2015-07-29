# -*- coding: utf-8 -*-
"""
Created on Fri Jun 05 11:50:01 2015

@author: Aaron

TODO: Implement Geometry Checking
    - Test if any lixels overlap
    - Cannot have hole in a hole, or solid in a solid
TODO: Use Bounded Volume Heirarchy to Reduce Lixel Search
TODO: Account for air attenuation by including outer material
      Currently will break if need to account for two materials in contact
"""
from mesh import angle_matrix
from geometry import Geometry, line_segment_intersect, ray_segment_intersect
from material import Material
from detector import DetectorArc, DetectorPlane
import numpy as np
import matplotlib.pyplot as plt

class Simulation(object):
    def __init__(self, universe_material, diameter=100., detector_width=100., detector='plane'):
        self.universe_material = universe_material
        self.geometry = Geometry()
        self.source = np.array([-diameter / 2., 0.])
        if detector == 'plane':
            self.detector = DetectorPlane([diameter / 2., 0.], detector_width)
        elif detector == 'arc':
            self.detector = DetectorArc([diameter / 2., 0], diameter, detector_width / 2., -detector_width / 2.)

    def get_intersecting_lixels(self, start, end, ray=False):
        """
        Find intersection points and lixels which intersect ray (start, end).
        """
        intercepts, indexes = [], []
        segment = np.array([start, end])

        if ray:
            intersect_func = ray_segment_intersect
        else:
            intersect_func = line_segment_intersect

        for i, lixel in enumerate(self.geometry.mesh.lixels):
            intercept = intersect_func(self.geometry.mesh.points[lixel], segment)
            if intercept is not None:
                intercepts.append(intercept)
                indexes.append(i)

        return intercepts, indexes
   
    def attenuation_length(self, start, end):
        """
        Calculate attenuation length for geometry.

        Can account for starting and ending in any position.
        """
        intercepts, indexes = self.get_intersecting_lixels(start, end)

        if len(intercepts) == 0:
            ray_intercepts, ray_indexes = self.get_intersecting_lixels(start, end, ray=True)
            if len(ray_intercepts) == 0:
                return np.linalg.norm(start - end) * self.universe_material.attenuation

            distances = np.linalg.norm(np.add(ray_intercepts, -start), axis=1)
            distances_argmin = np.argmin(distances)
            closest_index = ray_indexes[distances_argmin]
            closest_intercept = ray_intercepts[distances_argmin]
            closest_normal = self.geometry.mesh.lixel_normal(closest_index)
            start_sign = np.sign(np.dot(start - closest_intercept, closest_normal))

            if start_sign > 0:
                outer_atten = self.geometry.materials[self.geometry.outer_material_index[closest_index]].attenuation
                atten_length = np.linalg.norm(start - end) * outer_atten
            else:
                inner_atten = self.geometry.materials[self.geometry.inner_material_index[closest_index]].attenuation
                atten_length = np.linalg.norm(start - end) * inner_atten

            return atten_length

        distances = np.linalg.norm(np.add(intercepts, -start), axis=1)
        distances_argmin = np.argmin(distances)
        closest_index = indexes[distances_argmin]
        closest_intercept = intercepts[distances_argmin]
        closest_normal = self.geometry.mesh.lixel_normal(closest_index)
        start_sign = np.sign(np.dot(start - closest_intercept, closest_normal))

        if start_sign > 0:
            outer_atten = self.geometry.materials[self.geometry.outer_material_index[closest_index]].attenuation
            atten_length = np.linalg.norm(start - end) * outer_atten
        else:
            inner_atten = self.geometry.materials[self.geometry.inner_material_index[closest_index]].attenuation
            atten_length = np.linalg.norm(start - end) * inner_atten

        for intercept, index in zip(intercepts, indexes):
            normal = self.geometry.mesh.lixel_normal(index)
            start_sign = np.sign(np.dot(start - intercept, normal))
            end_sign = np.sign(np.dot(end - intercept, normal))
            inner_atten = self.geometry.materials[self.geometry.inner_material_index[index]].attenuation
            outer_atten = self.geometry.materials[self.geometry.outer_material_index[index]].attenuation

            atten_length += start_sign * np.linalg.norm(intercept - end) * (inner_atten - outer_atten)
        
        return atten_length

    # TODO : Account for joined fissionable materials
    def fission_segments(self, start, end):
        segment_points = []
        for i, lixel in enumerate(self.geometry.mesh.lixels):
            intercept = line_segment_intersect(self.geometry.mesh.points[lixel], np.array([start, end]))
            if intercept is not None:
                inner_material = self.geometry.materials[self.geometry.inner_material_index[i]]
                outer_material = self.geometry.materials[self.geometry.outer_material_index[i]]
                normal = self.geometry.mesh.lixel_normal(i)
                sign = np.sign(np.dot(start - intercept, normal))
                if inner_material.is_fissionable or outer_material.is_fissionable:
                    segment_points.append([intercept, sign, inner_material.macro_fission, outer_material.macro_fission])

        # calculate segments
        distances = [np.sqrt((s[0][0] - start[0]) ** 2. + (s[0][1] - start[1]) ** 2.) for s in segment_points]
        segment_points_order = [index for (distance, index) in sorted(zip(distances, range(len(distances))))]
        ordered_segment_points = [segment_points[i] for i in segment_points_order]
        
        if len(ordered_segment_points) % 2 != 0:
            raise IndexError
        
        # TODO : Fix by accounting for fissionable materials on either side
        # not correct, but should be okay for simple cases
        start_point = [ordered_segment_points[i][0] for i in xrange(0, len(ordered_segment_points), 2)]
        end_point = [ordered_segment_points[i][0] for i in xrange(1, len(ordered_segment_points), 2)]
        macro_fission = [ordered_segment_points[i][2] for i in xrange(0, len(ordered_segment_points), 2)] 

        return start_point, end_point, macro_fission
    
    def scan(self, angles=[0], nbins=100):
        atten_length = np.zeros((nbins, len(angles)))
        
        detector_bins = self.detector.create_bins(nbins)
        source = self.source
        
        for i, angle in enumerate(angles):
            rot = angle_matrix(angle)
            rot_source = np.inner(source, rot)
            rot_detector_bins = np.inner(detector_bins, rot)
            for j, detector_bin in enumerate(rot_detector_bins):
                atten_length[j, i] = self.attenuation_length(rot_source, detector_bin)
        
        return atten_length
    
    def radon_transform(self, angles=[0], nbins=100):
        if type(self.detector) is not DetectorPlane:
            raise TypeError('self.detector is not DetectorPlane')
        radon = np.zeros((nbins, len(angles)))
        
        detector_bins = self.detector.create_bins(nbins)
        source_bins = np.inner(detector_bins, angle_matrix(180.))[::-1]
        
        for i, angle in enumerate(angles):
            rot = angle_matrix(angle)
            rot_source = np.inner(source_bins, rot)
            rot_detector = np.inner(detector_bins, rot)
            for j in xrange(len(rot_detector)):
                radon[j, i] = self.attenuation_length(rot_source[j], rot_detector[j])
        
        return radon

    def draw(self, draw_normals=False):
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

        if draw_normals:
            for solid in self.geometry.solids:
                for i, lixel in enumerate(solid.mesh.lixels):
                    points = solid.mesh.points[lixel]
                    centroid_x = (points[0, 0] + points[1, 0]) / 2.
                    centroid_y = (points[0, 1] + points[1, 1]) / 2.
                    normal = solid.mesh.lixel_normal(i)
                    plt.arrow(centroid_x, centroid_y, normal[0], normal[1], width=0.01, fc=solid.color, ec=solid.color)
        
        if self.source is not None:
            plt.scatter(self.source[0], self.source[1], color='red', marker='x')
        
        detector_bins = self.detector.create_bins()
        plt.plot(detector_bins[:, 0], detector_bins[:, 1], color='green')

        plt.axis('equal')
        plt.xlabel('X (cm)')
        plt.ylabel('Y (cm)')