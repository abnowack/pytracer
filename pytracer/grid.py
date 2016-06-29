import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


# TODO: Organize file structure layout
# TODO: Breakout all algorithms into separate files, including fission stuff


class Grid(object):
    def __init__(self, width, height, nx, ny, origin=None, angle=0):
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
        x = np.linspace(-self.width / 2, self.width / 2, self.nx + 1)
        y = np.linspace(self.height / 2, -self.height / 2, self.ny + 1)

        points = np.zeros((self.ny + 1, self.nx + 1, 2))
        points[:, :, 0] = x
        points[:, :, 1] = y[:, np.newaxis]

        rotation = math2d.angle_matrix(self.angle)
        points = np.dot(points, rotation)
        points += self.origin

        return points

    def rotate(self, angle=0):
        self.angle = angle
        self.points = self.render()

    def draw_points(self):
        plt.scatter(self.points[:, :, 0], self.points[:, :, 1], zorder=10)

    def draw_lines(self):
        # draw horizontal lines
        start, end = self.points[:, 0], self.points[:, -1]
        for i in range(len(start)):
            plt.plot([start[i, 0], end[i, 0]], [start[i, 1], end[i, 1]], color='black', zorder=10)

        # draw vertical lines
        start, end = self.points[0, :], self.points[-1, :]
        for i in range(self.nx + 1):
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
        ix, iy = cell_i % self.nx, cell_i / self.nx
        p1 = self.points[iy, ix]
        p2 = self.points[iy + 1, ix]
        p3 = self.points[iy + 1, ix + 1]
        p4 = self.points[iy, ix + 1]
        return np.array([p1, p2, p3, p4, p1])

    def raster_points(self, i):
        rpoints = np.array([[-2., -6.], [6., -2.], [-6., 2.], [2., 6.]]) / 10.
        rotmat = math2d.angle_matrix(self.angle)
        rpoints = np.dot(rpoints, rotmat)
        rpoints += np.average(self.cell_boundary(i), axis=0)

        return rpoints

    def create_mesh(self, cell_i):
        points = self.cell_boundary(cell_i)
        return Mesh(math2d.create_segments_from_points(points))

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
    detector_solid_angle = math2d.solid_angle(sim.detector.segments, point) / (2 * np.pi)  # returns 200,200
    in_attenuation_length = sim.attenuation_length(sim.source.pos, point)
    segment_centers = math2d.center(sim.detector.segments)
    out_attenuation_lengths = np.array([sim.attenuation_length(point, center) for center in segment_centers])

    prob = np.exp(-in_attenuation_length) * np.multiply(detector_solid_angle, np.exp(-out_attenuation_lengths))

    return prob
