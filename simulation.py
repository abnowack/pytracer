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
            self.detector = DetectorArc(self.source, diameter, detector_width / 2., -detector_width / 2.)

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
                outer_atten = self.geometry.get_outer_material(closest_index).attenuation
                atten_length = np.linalg.norm(start - end) * outer_atten
            else:
                inner_atten = self.geometry.get_inner_material(closest_index).attenuation
                atten_length = np.linalg.norm(start - end) * inner_atten

            return atten_length

        distances = np.linalg.norm(np.add(intercepts, -start), axis=1)
        distances_argmin = np.argmin(distances)
        closest_index = indexes[distances_argmin]
        closest_intercept = intercepts[distances_argmin]
        closest_normal = self.geometry.mesh.lixel_normal(closest_index)
        start_sign = np.sign(np.dot(start - closest_intercept, closest_normal))

        if start_sign > 0:
            outer_atten = self.geometry.get_outer_material(closest_index).attenuation
            atten_length = np.linalg.norm(start - end) * outer_atten
        else:
            inner_atten = self.geometry.get_inner_material(closest_index).attenuation
            atten_length = np.linalg.norm(start - end) * inner_atten

        for intercept, index in zip(intercepts, indexes):
            normal = self.geometry.mesh.lixel_normal(index)
            start_sign = np.sign(np.dot(start - intercept, normal))
            end_sign = np.sign(np.dot(end - intercept, normal))
            inner_atten = self.geometry.get_inner_material(index).attenuation
            outer_atten = self.geometry.get_outer_material(index).attenuation

            atten_length += start_sign * np.linalg.norm(intercept - end) * (inner_atten - outer_atten)
        
        return atten_length

    def fission_segments(self, start, end):
        """
        Return list of line segments where fissions may occur.
        """
        segments, cross_sections = [], []

        intercepts, indexes = self.get_intersecting_lixels(start, end)

        # otherwise
        fission_indexes = []
        fission_intercepts = []
        for index, intercept in zip(indexes, intercepts):
            # test if lixel is border of fissionable material(s)
            inner_material = self.geometry.get_inner_material(index)
            outer_material = self.geometry.get_outer_material(index)
            if inner_material.is_fissionable or outer_material.is_fissionable:
                fission_indexes.append(index)
                fission_intercepts.append(intercept)

        # account for no intersections with fissionable materials
        if len(fission_intercepts) == 0:
            return segments, cross_sections

        # sort fission_indexes and fission_intercepts by distance from start
        distances = np.linalg.norm(np.add(fission_intercepts, -start), axis=1)
        distance_order = [index_ for (distance_, index_) in sorted(zip(distances, range(len(distances))))]

        sorted_fission_indexes = [fission_indexes[i] for i in distance_order]
        sorted_fission_intercepts = [fission_intercepts[i] for i in distance_order]

        for i in xrange(len(sorted_fission_indexes)):
            f_ind = sorted_fission_indexes[i]
            f_int = sorted_fission_intercepts[i]

            if i == 0:
                # test if start to first fission lixel is fissionable
                normal = self.geometry.mesh.lixel_normal(f_ind)
                sign = np.sign(np.dot(start - f_int, normal))

                inner_material = self.geometry.get_inner_material(f_ind)
                outer_material = self.geometry.get_outer_material(f_ind)

                if sign > 0 and outer_material.is_fissionable:
                    segments.append([start, f_int])
                    cross_sections.append(outer_material.macro_fission)
                elif sign < 0 and inner_material.is_fissionable:
                    segments.append([start, f_int])
                    cross_sections.append(inner_material.macro_fission)
            elif i == len(sorted_fission_indexes)-1:
                # test if end to last fission lixel is fissionable
                normal = self.geometry.mesh.lixel_normal(f_ind)
                sign = np.sign(np.dot(end - f_int, normal))

                inner_material = self.geometry.get_inner_material(f_ind)
                outer_material = self.geometry.get_outer_material(f_ind)

                if sign > 0 and outer_material.is_fissionable:
                    segments.append([f_int, end])
                    cross_sections.append(outer_material.macro_fission)
                elif sign < 0 and inner_material.is_fissionable:
                    segments.append([f_int, end])
                    cross_sections.append(inner_material.macro_fission)
                continue

            # test all intervening segments
            normal_1 = self.geometry.mesh.lixel_normal(f_ind)
            sign_1 = np.sign(np.dot(start - f_int, normal_1))

            f_ind2 = sorted_fission_indexes[i+1]
            f_int2 = sorted_fission_intercepts[i+1]

            normal_2 = self.geometry.mesh.lixel_normal(f_ind2)
            sign_2 = np.sign(np.dot(start - f_int2, normal_2))

            if sign_1 > 0:
                mat_1 = self.geometry.get_inner_material(f_ind)
            else:
                mat_1 = self.geometry.get_outer_material(f_ind)

            if sign_2 < 0:
                mat_2 = self.geometry.get_inner_material(f_ind)
            else:
                mat_2 = self.geometry.get_outer_material(f_ind)

            if mat_1.is_fissionable and mat_2.is_fissionable:
                if mat_1.macro_fission != mat_2.macro_fission:
                    raise NotImplementedError
                segments.append([f_int, f_int2])
                cross_sections.append(mat_1.macro_fission)

        return segments, cross_sections
    
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
        self.geometry.draw(draw_normals)
        
        if self.source is not None:
            plt.scatter(self.source[0], self.source[1], color='red', marker='x')
        
        self.detector.draw(draw_normals)

        plt.axis('equal')
        plt.xlabel('X (cm)')
        plt.ylabel('Y (cm)')