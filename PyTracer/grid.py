import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

import math2d
from scripts.raytrace import build_shielded_geometry


# TODO: Organize file structure layout
# TODO: Breakout all algorithms into separate files, including fission stuff

class Grid(object):
    def __init__(self, width, height, nx, ny, origin=None, angle=0.):
        self.width = width
        self.height = height
        self.nx = nx
        self.ny = ny
        if origin is None:
            self.origin = np.array([0., 0.])
        else:
            self.origin = np.array(origin)
        self.angle = angle

        self.points = self.render()

    def render(self):
        x = np.linspace(-self.width/2., self.width/2., self.nx+1)
        y = np.linspace(self.height/2., -self.height/2., self.ny+1)

        points = np.zeros((self.nx+1, self.ny+1, 2))
        points[:, :, 0] = x
        points[:, :, 1] = y[:, np.newaxis]

        rotation = math2d.angle_matrix(self.angle)
        points = np.dot(points, rotation)
        points += self.origin

        return points

    def rotate(self, angle=0.):
        self.angle = angle
        self.points = self.render()

    def draw_points(self):
        plt.scatter(self.points[:, :, 0], self.points[:, :, 1], zorder=10)

    def draw_lines(self):
        # draw horizontal lines
        start, end = self.points[:, 0], self.points[:, -1]
        for i in xrange(len(start)):
            plt.plot([start[i, 0], end[i, 0]], [start[i, 1], end[i, 1]], color='black', zorder=10)

        # draw vertical lines
        start, end = self.points[0, :], self.points[-1, :]
        for i in xrange(self.nx+1):
            plt.plot([start[i, 0], end[i, 0]], [start[i, 1], end[i, 1]], color='black', zorder=10)

    def draw_cell(self, i):
        ax = plt.gca()
        ax.add_patch(Polygon(self.cell_boundary(i), facecolor="red", alpha=0.5))

    def draw_raster_samples(self, i):
        rpoints = self.raster_points(i)
        plt.scatter(rpoints[:, 0], rpoints[:, 1], zorder=11)

    @property
    def ncells(self):
        return self.nx * self.ny

    def cell_boundary(self, cell_i):
        ix, iy = cell_i % self.nx, cell_i / self.ny
        p1 = self.points[ix, iy]
        p2 = self.points[ix+1, iy]
        p3 = self.points[ix+1, iy+1]
        p4 = self.points[ix, iy+1]
        return np.array([p1, p2, p3, p4])

    def raster_points(self, i):
        rpoints = np.array([[-2., -6.], [6., -2.], [-6., 2.], [2., 6.]]) / 10.
        rotmat = math2d.angle_matrix(self.angle)
        rpoints = np.dot(rpoints, rotmat)
        rpoints += np.average(self.cell_boundary(i), axis=0)

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
    in_attenuation_length = sim.attenuation_length(sim.source.pos, point)
    segment_centers = math2d.center(sim.detector.segments)
    out_attenuation_lengths = np.array([sim.attenuation_length(point, center) for center in segment_centers])

    prob = np.exp(-in_attenuation_length) * np.multiply(detector_solid_angle, np.exp(-out_attenuation_lengths))

    return prob


def main():
    sim = build_shielded_geometry(True)
    sim.grid = Grid(20, 20, 10, 10)
    sim.rotate(10.)

    plt.figure()
    sim.draw(False)
    sim.grid.draw_cell(16)
    sim.grid.draw_raster_samples(0)

    # plt.figure()

    # Grid in place, plot each cell
    # for i in xrange(sim.grid.ncells):
    #     print i, sim.grid.ncells
    #     if i == 0:
    #         p = sim.grid.cell_prob(i, sim)
    #         p_matrix = np.zeros((sim.grid.ncells, len(p)))
    #         p_matrix[0] = p[:]
    #     else:
    #         p_matrix[i] = sim.grid.cell_prob(i, sim)

    # Cell in place, rotate cell
    # nangles, cell_i = 100, 14
    # angles = np.linspace(0., 180., nangles)
    # for i, angle in enumerate(angles):
    #     print i, nangles
    #     sim.rotate(angle)
    #     if i == 0:
    #         p = sim.grid.cell_prob(cell_i, sim)
    #         p_matrix = np.zeros((nangles, len(p)))
    #         p_matrix[0] = p[:]
    #     else:
    #         p_matrix[i] = sim.grid.cell_prob(cell_i, sim)
    #
    # plt.imshow(p_matrix.T, interpolation='none', aspect='auto')

    plt.show()

if __name__ == "__main__":
    sys.exit(int(main() or 0))