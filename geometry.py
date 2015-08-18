from mesh import Mesh
import numpy as np
import matplotlib.pyplot as plt

class Geometry(object):
    def __init__(self):
        self.solids = []
    
    def flatten(self):
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
        for solid in self.solids:
            solid.draw(draw_normals)