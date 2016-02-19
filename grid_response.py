import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from raytrace import build_shielded_geometry
import math2d


class Grid(object):
    def __init__(self, width, height, nx, ny, origin=[0., 0.], angle=0.):
        self.width = width
        self.height = height
        self.nx = nx
        self.ny = ny
        self.origin = origin
        self.angle = angle

        x = np.linspace(-width/2., width/2., nx+1)
        y = np.linspace(-height/2., height/2., ny+1)
        self.x, self.y = np.meshgrid(x, y)

    def drawpoints(self):
        plt.scatter(self.x, self.y, zorder=10)

    def drawlines(self):
        # draw horizontal lines
        for i in xrange(self.ny+1):
            x0, x1 = self.x[i, 0], self.x[i, -1]
            y0, y1 = self.y[i, 0], self.y[i, -1]
            plt.plot([x0, x1], [y0, y1], color='black', zorder=10)
        # draw vertical lines
        for i in xrange(self.nx+1):
            x0, x1 = self.x[0, i], self.x[-1, i]
            y0, y1 = self.y[0, i], self.y[-1, i]
            plt.plot([x0, x1], [y0, y1], color='black', zorder=10)

    def drawcell(self, i):
        points = self.cell_boundary(i)
        ax = plt.gca()
        ax.add_patch(Rectangle(points[0], self.width/self.nx, self.height/self.ny, facecolor="red"))

    def draw_rastpoints(self, i):
        rpoints = self.raster_points(i)
        plt.scatter(rpoints[:, 0], rpoints[:, 1], zorder=11)

    @property
    def ncells(self):
        return self.nx * self.ny

    def xy_index(self, i):
        return i / self.nx, i % self.ny

    def cell_boundary(self, cell_i):
        i, j = self.xy_index(cell_i)
        bounds = np.zeros((4, 2))

        bounds[0, 0], bounds[0, 1] = self.x[i, j], self.y[i, j]
        bounds[1, 0], bounds[1, 1] = self.x[i+1, j], self.y[i+1, j]
        bounds[2, 0], bounds[2, 1] = self.x[i, j+1], self.y[i, j+1]
        bounds[3, 0], bounds[3, 1] = self.x[i+1, j+1], self.y[i+1, j+1]

        return bounds

    def raster_points(self, i, rx=0.7, ry=0.9):
        points = self.cell_boundary(i)
        rpoints = np.zeros(points.shape)
        rpoints[0, 0], rpoints[0, 1] = rx, ry
        rpoints[1, 0], rpoints[1, 1] = ry, 1-rx
        rpoints[2, 0], rpoints[2, 1] = 1-rx, 1-ry
        rpoints[3, 0], rpoints[3, 1] = 1-ry, rx
        rpoints[:, 0] *= self.width/self.nx
        rpoints[:, 1] *= self.height/self.ny
        rpoints[:, 0] += points[0, 0]
        rpoints[:, 1] += points[0, 1]

        return rpoints

    def cell_prob(self, i, sim):
        rpoints = self.raster_points(i)
        prob1 = propagate_fissions_point_detector(sim, rpoints[0])
        prob2 = propagate_fissions_point_detector(sim, rpoints[1])
        prob3 = propagate_fissions_point_detector(sim, rpoints[2])
        prob4 = propagate_fissions_point_detector(sim, rpoints[3])

        return (prob1 + prob2 + prob3 + prob4) / 4.

def propagate_fissions_point_detector(sim, point):
    """
    Calculate probability of induced fission being detected over detector plane.

    nu = 1 for now, not using macro_fission
    """
    detector_solid_angle = math2d.solid_angle(sim.detector.segments, point) / (2. * np.pi) # returns 200,200
    in_attenuation_length = sim.attenuation_length(sim.source, point)
    segment_centers = math2d.center(sim.detector.segments)
    out_attenuation_lengths = np.array([sim.attenuation_length(point, center) for center in segment_centers])

    prob = np.exp(-in_attenuation_length) * np.multiply(detector_solid_angle, np.exp(-out_attenuation_lengths))

    return prob

def main():
    sim = build_shielded_geometry(True)
    grid = Grid(20, 20, 10, 10)

    print grid.cell_boundary(0)
    print grid.raster_points(0)

    plt.figure()
    sim.draw(False)
    grid.drawlines()
    # grid.drawcell(2)
    # grid.draw_rastpoints(2)
    #
    # plt.figure()
    #
    # for i in xrange(25):
    #     if i == 0:
    #         p = grid.cell_prob(i, sim)
    #         p_matrix = np.zeros((25, len(p)))
    #         p_matrix[0] = p[:]
    #     else:
    #         p_matrix[i] = grid.cell_prob(i, sim)
    #
    # plt.imshow(p_matrix.T, interpolation='none', aspect='auto')

    plt.show()

if __name__ == "__main__":
    sys.exit(int(main() or 0))