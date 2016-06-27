import numpy as np
from . import math2d, math2d_c
from .mesh import Mesh
from .material import Material


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

        self.intersects_cache = np.empty((100, 2), dtype=np.double)
        self.indexes_cache = np.empty(100, dtype=np.int)

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

    # TODO memoize?
    @property
    def inner_attenuation(self):
        inner_atten = np.zeros(len(self.inner_materials))
        for i, inner_mat in enumerate(self.inner_materials):
            inner_atten[i] = self.inner_materials[i].attenuation
        return inner_atten

    # TODO memoize?
    @property
    def outer_attenuation(self):
        outer_atten = np.zeros(len(self.outer_materials))
        for i, outer_mat in enumerate(self.outer_materials):
            outer_atten[i] = self.outer_materials[i].attenuation
        return outer_atten

    def get_intersecting_segments(self, start, end, ray=False):
        n_intersects = math2d_c.intersections(self.mesh.segments, np.array([start, end]), self.intersects_cache,
                                              self.indexes_cache, ray)
        return self.intersects_cache[:n_intersects], self.indexes_cache[:n_intersects]

    def attenuation_length(self, start, end=None, attenuation_cache=None):
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
        if end is not None:
            attenuation = math2d_c.calc_attenuation(self.mesh.segments, np.array([start, end]), self.inner_attenuation,
                                                    self.outer_attenuation, self.universe_material.attenuation,
                                                    self.intersects_cache, self.indexes_cache)
            return attenuation
        else:
            attenuations = math2d_c.calc_attenuation_bulk(self.mesh.segments, start,
                                                          self.inner_attenuation, self.outer_attenuation,
                                                          self.universe_material.attenuation,
                                                          self.intersects_cache, self.indexes_cache, attenuation_cache)

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
        distance_order = [index_ for (distance_, index_) in sorted(zip(distances, list(range(len(distances)))))]

        sorted_fission_indexes = [fission_indexes[i] for i in distance_order]
        sorted_fission_intercepts = [fission_intercepts[i] for i in distance_order]

        for i in range(len(sorted_fission_indexes)):
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
            elif i == len(sorted_fission_indexes) - 1:
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

            f_ind2 = sorted_fission_indexes[i + 1]
            f_int2 = sorted_fission_intercepts[i + 1]

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
