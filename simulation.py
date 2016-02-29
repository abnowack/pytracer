from geometry import Geometry
from material import Material
from detector import DetectorArc, DetectorPlane
from particle_source import Source
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
        nbins : int
            number of segments for detector
        diameter : float
            Diameter of simulation, distance of source to detector plane
        detector_width : float
            Width of detector
        detector : str
            Type of detector ('arc', 'plane')
        """
        self.universe_material = universe_material
        self.geometry = Geometry()
        self.source = None
        self.detector = None
        self.grid = None

    def add_aligned_source_detector(self, diameter=100., nbins=100, width=100., type='plane'):
        self.source = Source(-diameter / 2., 0.)
        if type == 'plane':
            self.detector = DetectorPlane([diameter / 2., 0.], width, nbins)
        elif type == 'arc':
            self.detector = DetectorArc(self.source.pos, diameter, width / 2., -width / 2., nbins)

    def get_intersecting_segments(self, start, end, ray=False):
        """
        Calculate which segments in geometry intersect the line defined by [start, end]

        Parameters
        ----------
        start : (2) ndarray
            Start point of intersect line
        end : (2) ndarray
            End point of intersect line
        ray : bool
            Indicate whether intersect line is a ray starting from `start`

        Returns
        -------
        intercepts : list
            list of intercept points [point_x, point_y]
        indexes : list
            index of intersecting segments in geometry class

        """
        intercepts, indexes = [], []
        intersect_segment = np.array([start, end])

        for i, segment in enumerate(self.geometry.mesh.segments):
            intercept = math2d.intersect(segment, intersect_segment, ray)
            if intercept is not None:
                intercepts.append(intercept)
                indexes.append(i)

        return intercepts, indexes
   
    def attenuation_length(self, start, end):
        """
        Calculate of attenuation length through geometry from start to end

        Parameters
        ----------
        start : (2) array_like
            start position
        end : (2) array_like
            end positions

        Returns
        -------
        atten_length : float
            calculated attenuation length
        """
        intercepts, indexes = self.get_intersecting_segments(start, end)

        if len(intercepts) == 0:
            ray_intercepts, ray_indexes = self.get_intersecting_segments(start, end, ray=True)
            if len(ray_intercepts) == 0:
                return np.linalg.norm(start - end) * self.universe_material.attenuation

            distances = np.linalg.norm(np.add(ray_intercepts, -start), axis=1)
            distances_argmin = np.argmin(distances)
            closest_index = ray_indexes[distances_argmin]
            closest_intercept = ray_intercepts[distances_argmin]
            closest_normal = math2d.normal(self.geometry.mesh.segments[closest_index])
            start_sign = np.sign(np.dot(start - closest_intercept, closest_normal))

            if start_sign > 0:
                outer_atten = self.geometry.outer_materials[closest_index].attenuation
                atten_length = np.linalg.norm(start - end) * outer_atten
            else:
                inner_atten = self.geometry.inner_materials[closest_index].attenuation
                atten_length = np.linalg.norm(start - end) * inner_atten

            return atten_length

        distances = np.linalg.norm(np.add(intercepts, -start), axis=1)
        distances_argmin = np.argmin(distances)
        closest_index = indexes[distances_argmin]
        closest_intercept = intercepts[distances_argmin]
        closest_normal = math2d.normal(self.geometry.mesh.segments[closest_index])
        start_sign = np.sign(np.dot(start - closest_intercept, closest_normal))

        if start_sign > 0:
            outer_atten = self.geometry.outer_materials[closest_index].attenuation
            atten_length = np.linalg.norm(start - end) * outer_atten
        else:
            inner_atten = self.geometry.inner_materials[closest_index].attenuation
            atten_length = np.linalg.norm(start - end) * inner_atten

        for intercept, index in zip(intercepts, indexes):
            normal = math2d.normal(self.geometry.mesh.segments[index])
            start_sign = np.sign(np.dot(start - intercept, normal))
            inner_atten = self.geometry.inner_materials[index].attenuation
            outer_atten = self.geometry.outer_materials[index].attenuation

            atten_length += start_sign * np.linalg.norm(intercept - end) * (inner_atten - outer_atten)
        
        return atten_length

    def fission_segments(self, start, end):
        """
        Return list of line segments which traverse fissionable material

        Parameters
        ----------
        start
        end

        Returns
        -------

        """
        segments, cross_sections = [], []

        intercepts, indexes = self.get_intersecting_segments(start, end)

        # otherwise
        fission_indexes = []
        fission_intercepts = []
        for index, intercept in zip(indexes, intercepts):
            # test if lixel is border of fissionable material(s)
            inner_material = self.geometry.inner_materials[index]
            outer_material = self.geometry.outer_materials[index]
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
                normal = math2d.normal(self.geometry.mesh.segments[f_ind])
                sign = np.sign(np.dot(start - f_int, normal))

                inner_material = self.geometry.inner_materials[f_ind]
                outer_material = self.geometry.outer_materials[f_ind]

                if sign > 0 and outer_material.is_fissionable:
                    segments.append([start, f_int])
                    cross_sections.append(outer_material.macro_fission)
                elif sign < 0 and inner_material.is_fissionable:
                    segments.append([start, f_int])
                    cross_sections.append(inner_material.macro_fission)
            elif i == len(sorted_fission_indexes)-1:
                # test if end to last fission lixel is fissionable
                normal = math2d.normal(self.geometry.mesh.segments[f_ind])
                sign = np.sign(np.dot(end - f_int, normal))

                inner_material = self.geometry.inner_materials[f_ind]
                outer_material = self.geometry.outer_materials[f_ind]

                if sign > 0 and outer_material.is_fissionable:
                    segments.append([f_int, end])
                    cross_sections.append(outer_material.macro_fission)
                elif sign < 0 and inner_material.is_fissionable:
                    segments.append([f_int, end])
                    cross_sections.append(inner_material.macro_fission)
                continue

            # test all intervening segments
            normal_1 = math2d.normal(self.geometry.mesh.segments[f_ind])
            sign_1 = np.sign(np.dot(start - f_int, normal_1))

            f_ind2 = sorted_fission_indexes[i+1]
            f_int2 = sorted_fission_intercepts[i+1]

            normal_2 = math2d.normal(self.geometry.mesh.segments[f_ind2])
            sign_2 = np.sign(np.dot(start - f_int2, normal_2))

            if sign_1 > 0:
                mat_1 = self.geometry.inner_materials[f_ind]
            else:
                mat_1 = self.geometry.outer_materials[f_ind]

            if sign_2 < 0:
                mat_2 = self.geometry.inner_materials[f_ind]
            else:
                mat_2 = self.geometry.outer_materials[f_ind]

            if mat_1.is_fissionable and mat_2.is_fissionable:
                if mat_1.macro_fission != mat_2.macro_fission:
                    raise NotImplementedError
                segments.append([f_int, f_int2])
                cross_sections.append(mat_1.macro_fission)

        return segments, cross_sections
    
    def scan(self, angles=[0], nbins=100):
        atten_length = np.zeros((nbins, len(angles)))
        
        self.detector.nbins = nbins
        self.detector.render()
        
        for i, angle in enumerate(angles):
            self.rotate(angle)
            detector_centers = math2d.center(self.detector.segments)
            for j, detector_center in enumerate(detector_centers):
                atten_length[j, i] = self.attenuation_length(self.source.pos, detector_center)
        
        return atten_length

    def rotate(self, angle):
        self.detector.rotate(angle)
        self.source.rotate(angle)
        self.grid.rotate(angle)
    
    def radon_transform(self, angles=[0]):
        if type(self.detector) is not DetectorPlane:
            raise TypeError('self.detector is not DetectorPlane')

        radon = np.zeros((np.size(self.detector.nbins-1, 0), len(angles)))

        for i, angle in enumerate(angles):
            self.rotate(angle)
            detector_points = math2d.center(self.detector.segments)
            source_points = np.dot(detector_points, math2d.angle_matrix(180.))[::-1]
            for j, (source_point, detector_point) in enumerate(izip(detector_points, source_points)):
                radon[j, i] = self.attenuation_length(source_point, detector_point)
        
        return radon

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