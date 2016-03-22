import matplotlib.pyplot as plt
from PyTracer.detector import DetectorArc, DetectorPlane
from PyTracer.geometry import Geometry
from source import Source


class Simulation(object):
    def __init__(self, universe_material):
        """
        Coordinate between Geometry (meshes) and detector plane

        Parameters
        ----------
        universe_material : Material
            Default material, such as vacuum or air
        """
        self.geometry = Geometry(universe_material)
        self.source = None
        self.detector = None
        self.grid = None

    def add_aligned_source_detector(self, diameter=100., nbins=100, width=100., type='plane'):
        self.source = Source(-diameter / 2., 0.)
        if type == 'plane':
            self.detector = DetectorPlane([diameter / 2., 0.], width, nbins)
        elif type == 'arc':
            self.detector = DetectorArc(self.source.pos, diameter, width / 2., -width / 2., nbins)

    def rotate(self, angle):
        if self.detector:
            self.detector.rotate(angle)
        if self.source:
            self.source.rotate(angle)
        if self.grid:
            self.grid.rotate(angle)

    def draw(self, draw_normals=False):
        plt.figure(figsize=(7, 7))
        self.geometry.draw(draw_normals)

        if self.source is not None:
            plt.scatter(self.source.pos[0], self.source.pos[1], color='red', marker='x')

        if self.detector is not None:
            self.detector.draw(draw_normals)

        if self.grid is not None:
            self.grid.draw_lines()

        plt.xlabel('X (cm)')
        plt.ylabel('Y (cm)')
        max_lim = max(max(plt.xlim()), max(plt.ylim()))
        plt.xlim(-max_lim, max_lim)
        plt.ylim(-max_lim, max_lim)
        plt.axes().set_aspect('equal', 'datalim')
