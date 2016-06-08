import numpy as np

import math2d
from mesh import Mesh
from material import Material
from itertools import izip

from intersect_module import intersect_c, intersecting_segments_c

class Geometry(object):
    """
    Contains all mesh objects in the simulation, then translates geometry into simple arrays for fast computation
    """
    def __init__(self, universe_material=None):
        self.solids = []
        self.mesh = None
        self.inner_materials = None
        self.outer_materials = None
        if not universe_material:
            self.universe_material = universe_material
        else:
            self.universe_material = Material(0.00, color='white')

    def draw(self, draw_normals=False):
        """
        Draw all meshes

        Parameters
        ----------
        draw_normals : bool
            If true, also draw all normal vectors for each segment in each mesh
        """
        for solid in self.solids:
            solid.draw(draw_normals)
    
    def flatten(self):
        """
        Combine all meshes into a single mesh, and all inner and outer mesh materials into single material lists
        """
        for i, solid in enumerate(self.solids):
            if i == 0:
                self.mesh = Mesh(solid.mesh.segments)
                self.inner_materials = solid.inner_materials
                self.outer_materials = solid.outer_materials
            else:
                self.mesh += solid.mesh
                self.inner_materials = np.concatenate((self.inner_materials, solid.inner_materials))
                self.outer_materials = np.concatenate((self.outer_materials, solid.outer_materials))

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

        # Old Method 1
        # intersect_segment = np.array([start, end])
        # intercepts, indexes = [], []
        #
        # for i, segment in enumerate(self.mesh.segments):
        #     intercept = math2d.intersect(segment, intersect_segment, ray)
        #     if intercept is not None:
        #         intercepts.append(np.array(intercept))
        #         indexes.append(i)

        # Old Method 2, inline intercepts
        # intersect_segment = np.array([start, end])
        # intercepts, indexes = math2d.intersects(self.mesh.segments, intersect_segment, ray)

        # Inner C Loop Method 3
        # intersect_segment = np.array([start, end])
        # intercepts, indexes = [], []
        #
        # for i, segment in enumerate(self.mesh.segments):
        #     intercept = intersect_c(segment[0][0], segment[0][1], segment[1][0], segment[1][1],
        #                             start[0], start[1], end[0], end[1])
        #     if intercept is not None:
        #         intercepts.append(np.array(intercept))
        #         indexes.append(i)

        # Full C Implementation 4
        indexes, xcoords, ycoords = intersecting_segments_c(self.mesh.segments, start[0], start[1], end[0], end[1])
        intercepts = np.vstack((xcoords, ycoords)).T

        # Cythonized Version 5
        # intersect_segment = np.array([start, end])
        # intercepts, indexes = [], []
        #
        # for i, segment in enumerate(self.mesh.segments):
        #     intercept = intersect_cython(segment[0][0], segment[0][1], segment[1][0], segment[1][1],
        #                                  start[0], start[1], end[0], end[1])
        #     if intercept is not None:
        #         intercepts.append(np.array(intercept))
        #         indexes.append(i)

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
        no_segment_intercepts = False

        # If no intersection must determine what material we are within by tracing a ray
        if len(intercepts) == 0:
            intercepts, indexes = self.get_intersecting_segments(start, end, ray=True)
            no_segment_intercepts = True
        # No intersection through a ray, must be outside the object, return atten_length from universe material
        if len(intercepts) == 0:
            return np.linalg.norm(start - end) * self.universe_material.attenuation

        distances = np.linalg.norm(np.add(intercepts, -start), axis=1)
        distances_argmin = np.argmin(distances)
        closest_index = indexes[distances_argmin]
        closest_intercept = intercepts[distances_argmin]
        closest_normal = math2d.normal(self.mesh.segments[closest_index])
        start_sign = np.sign(np.dot(start - closest_intercept, closest_normal))

        if start_sign > 0:
            outer_atten = self.outer_materials[closest_index].attenuation
            atten_length = np.linalg.norm(start - end) * outer_atten
        else:
            inner_atten = self.inner_materials[closest_index].attenuation
            atten_length = np.linalg.norm(start - end) * inner_atten

        # No segment intercept, so return the beginning to end atten_length
        if no_segment_intercepts:
            return atten_length

        # Had intersections, so add up all individual atten_lengths between start to end
        for intercept, index in izip(intercepts, indexes):
            normal = math2d.normal(self.mesh.segments[index])
            start_sign = np.sign(np.dot(start - intercept, normal))
            inner_atten = self.inner_materials[index].attenuation
            outer_atten = self.outer_materials[index].attenuation

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
            inner_material = self.inner_materials[index]
            outer_material = self.outer_materials[index]
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
                normal = math2d.normal(self.mesh.segments[f_ind])
                sign = np.sign(np.dot(start - f_int, normal))

                inner_material = self.inner_materials[f_ind]
                outer_material = self.outer_materials[f_ind]

                if sign > 0 and outer_material.is_fissionable:
                    segments.append([start, f_int])
                    cross_sections.append(outer_material.macro_fission)
                elif sign < 0 and inner_material.is_fissionable:
                    segments.append([start, f_int])
                    cross_sections.append(inner_material.macro_fission)
            elif i == len(sorted_fission_indexes)-1:
                # test if end to last fission lixel is fissionable
                normal = math2d.normal(self.mesh.segments[f_ind])
                sign = np.sign(np.dot(end - f_int, normal))

                inner_material = self.inner_materials[f_ind]
                outer_material = self.outer_materials[f_ind]

                if sign > 0 and outer_material.is_fissionable:
                    segments.append([f_int, end])
                    cross_sections.append(outer_material.macro_fission)
                elif sign < 0 and inner_material.is_fissionable:
                    segments.append([f_int, end])
                    cross_sections.append(inner_material.macro_fission)
                continue

            # test all intervening segments
            normal_1 = math2d.normal(self.mesh.segments[f_ind])
            sign_1 = np.sign(np.dot(start - f_int, normal_1))

            f_ind2 = sorted_fission_indexes[i+1]
            f_int2 = sorted_fission_intercepts[i+1]

            normal_2 = math2d.normal(self.mesh.segments[f_ind2])
            sign_2 = np.sign(np.dot(start - f_int2, normal_2))

            if sign_1 > 0:
                mat_1 = self.inner_materials[f_ind]
            else:
                mat_1 = self.outer_materials[f_ind]

            if sign_2 < 0:
                mat_2 = self.inner_materials[f_ind]
            else:
                mat_2 = self.outer_materials[f_ind]

            if mat_1.is_fissionable and mat_2.is_fissionable:
                if mat_1.macro_fission != mat_2.macro_fission:
                    raise NotImplementedError
                segments.append([f_int, f_int2])
                cross_sections.append(mat_1.macro_fission)

        return segments, cross_sections