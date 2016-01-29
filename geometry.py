from mesh import Mesh
import numpy as np


class Geometry(object):
    """
    Contains all mesh objects in the simulation, then translates geometry into simple arrays for fast computation
    """
    def __init__(self):
        self.solids = []
        self.mesh = None
        self.inner_materials = None
        self.outer_materials = None
    
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