import numpy as np
import matplotlib.pyplot as plt

class Solid(object):
    def __init__(self, mesh, inner_material, outer_material):
        self.mesh = mesh
        self.inner_material = np.tile(inner_material, np.size(self.mesh.lixels, 0))
        self.outer_material = np.tile(outer_material, np.size(self.mesh.lixels, 0))
        self.color = inner_material.color
    
    def draw(self, draw_normals=False):
        lixels, points = self.mesh.continuous_path_order()
        xs = [points[lixels[0, 0]][0]]
        ys = [points[lixels[0, 0]][1]]
        for i, lixel in enumerate(lixels):
            if points[lixel[0]][0] == xs[-1] and points[lixel[0]][1] == ys[-1]:
                xs.append(points[lixel[1]][0])
                ys.append(points[lixel[1]][1])
            else:
                xs.extend(points[lixel][:, 0])
                ys.extend(points[lixel][:, 1])

        plt.fill(xs, ys, color=self.color)

        if draw_normals:
            for i, lixel in enumerate(self.mesh.lixels):
                points = self.mesh.points[lixel]
                centroid_x = (points[0, 0] + points[1, 0]) / 2.
                centroid_y = (points[0, 1] + points[1, 1]) / 2.
                normal = self.mesh.lixel_normal(i)
                plt.arrow(centroid_x, centroid_y, normal[0], normal[1], width=0.01, fc=self.color, ec=self.color)