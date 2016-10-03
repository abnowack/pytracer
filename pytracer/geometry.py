import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
from . import geometry_c as geo_c

_array1D_cache = np.empty(300, dtype=np.double)


def center(segments):
    """Calculates midpoints from line segments."""
    return (segments[:, 0] + segments[:, 1]) / 2


def normal(segments, midpoint_origin=False):
    """Calculates a normal vector to segment."""
    normals = np.zeros((np.size(segments, 0), 2), dtype=segments.dtype)
    normals[:, 0] = -(segments[:, 1, 1] - segments[:, 0, 1])
    normals[:, 1] = segments[:, 1, 0] - segments[:, 0, 0]
    length = np.sqrt(normals[:, 0] ** 2 + normals[:, 1] ** 2)
    normals = normals / length[:, np.newaxis]

    if midpoint_origin:
        normals += center(segments)
    return normals


def solid_angle(segments, point, cache=None):
    """Calculates 2D solid angle of segments relative to a point."""
    if cache is None:
        n_segments = np.size(segments, 0)
        cache = _array1D_cache[:n_segments]
    # a = np.linalg.norm(segments[:, 0] - point, axis=1)
    # b = np.linalg.norm(segments[:, 1] - point, axis=1)
    # c = np.linalg.norm(segments[:, 0] - segments[:, 1], axis=1)
    #
    # num = a ** 2 + b ** 2 - c ** 2
    # denom = 2 * a * b
    # angle = np.arccos(np.abs(num / denom))
    # is_greater_halfpi = angle > np.pi / 2
    # angle[is_greater_halfpi] = np.pi - angle[is_greater_halfpi]
    #
    # return angle

    geo_c.solid_angle(segments, point, cache)

    return cache


def rotation_matrix(radian):
    """Calculates 2D rotation matrix from an angle in radians."""
    return np.array([[np.cos(radian), np.sin(radian)], [-np.sin(radian), np.cos(radian)]])


def convert_points_to_segments(points, circular=False):
    """Converts array of points into array of line segments."""
    if not circular:
        segments = np.zeros((len(points) - 1, 2, 2))
        segments[:, 0, 0] = points[:-1, 0]
        segments[:, 1, 0] = points[1:, 0]
        segments[:, 0, 1] = points[:-1, 1]
        segments[:, 1, 1] = points[1:, 1]
    else:
        segments = np.zeros((len(points), 2, 2))
        segments[:-1, 0, 0] = points[:-1, 0]
        segments[:-1, 1, 0] = points[1:, 0]
        segments[:-1, 0, 1] = points[:-1, 1]
        segments[:-1, 1, 1] = points[1:, 1]

        segments[-1, 0, 0] = points[-1, 0]
        segments[-1, 0, 1] = points[-1, 1]
        segments[-1, 1, 0] = points[0, 0]
        segments[-1, 1, 1] = points[0, 1]

    return segments


def reorder_contiguous_order(segments, out=None):
    """Reorder line segments such that the last point of one segment, is the first of the next."""
    new_indices = []
    next_index = 0

    for i, segment in enumerate(segments):
        if next_index not in new_indices:
            new_indices.append(next_index)
        else:
            others = np.setdiff1d(segments, segments[new_indices])
            next_index = np.argwhere(segments[others[0]] == segments)
            new_indices.append(next_index)

        next_index = np.argwhere(segments[new_indices[-1], 1] == segments[:, 0])

    return segments[new_indices]


def translate(segments, shift, out=None):
    """Translate line segments by (shift[0], shift[1]) amount."""
    segments[:, :, 0] += shift[0]
    segments[:, :, 1] += shift[1]


def rotate(segments, radian, pivot=None, out=None):
    """Rotate line segments by radian around the point (pivot)."""
    if pivot:
        translate(segments, -pivot)

    xpoints = segments[:, :, 0] * np.cos(radian) - segments[:, :, 1] * np.sin(radian)
    ypoints = segments[:, :, 0] * np.sin(radian) + segments[:, :, 1] * np.cos(radian)

    segments[:, :, 0] = xpoints
    segments[:, :, 1] = ypoints

    if pivot:
        translate(segments, -pivot)


def create_horizontal_line(width, num_segments):
    points = np.zeros((num_segments + 1, 2), dtype=np.float32)
    points[:, 0] = np.linspace(-width / 2, width / 2, len(points))
    return convert_points_to_segments(points)


def create_vertical_line(height, num_segments):
    points = np.zeros((num_segments + 1, 2), dtype=np.float32)
    points[:, 1] = np.linspace(-height / 2, height / 2, len(points))
    return convert_points_to_segments(points)


def create_arc(radius, begin_radian=0, end_radian=2 * np.pi, num_segments=20):
    radians = np.linspace(begin_radian, end_radian, num_segments + 1)
    points = np.zeros((num_segments + 1, 2), dtype=np.float32)
    points[:-1, 0] = np.cos(radians) * radius
    points[:-1, 1] = np.sin(radians) * radius
    return convert_points_to_segments(points)


def create_circle(radius, num_segments=20):
    radians = np.linspace(np.pi, -np.pi, num_segments + 1)[:-1]
    points = np.zeros((num_segments, 2), dtype=np.float32)
    points[:, 0] = np.cos(radians) * radius
    points[:, 1] = np.sin(radians) * radius
    return convert_points_to_segments(points, circular=True)


def create_rectangle(width, height):
    points = np.zeros((5, 2), dtype=np.float32)
    points[0] = [width / 2, height / 2]
    points[1] = [width / 2, -height / 2]
    points[2] = [-width / 2, -height / 2]
    points[3] = [-width / 2, height / 2]
    points[4] = [width / 2, height / 2]
    return convert_points_to_segments(points)


def create_hollow(outer_segments, inner_segments):
    inner_segments = np.fliplr(inner_segments)[::-1]
    return np.concatenate((outer_segments, inner_segments))


def parallel_beam_paths(height, num_projections, offset, radians, extent=False):
    scan_line = center(create_vertical_line(height, num_projections))
    scan_line[:, 0] += offset
    start, end = np.zeros((len(scan_line), len(radians), 2)), np.zeros((len(scan_line), len(radians), 2))

    for i, radian in enumerate(radians):
        rot = rotation_matrix(radian)
        start[:, i, 0] = scan_line[:, 0] * rot[0, 0] + scan_line[:, 1] * rot[0, 1]
        start[:, i, 1] = scan_line[:, 0] * rot[1, 0] + scan_line[:, 1] * rot[1, 1]
    end = -start[::-1]

    if extent:
        return start, end, [radians[0], radians[-1], -height / 2, height / 2]
    else:
        return start, end


def fan_beam_paths(diameter, arc_radians, radians, extent=False):
    start = np.zeros((np.size(arc_radians, 0), np.size(radians, 0), 2))
    end = np.zeros(start.shape)

    start[:, :, 0] = (np.cos(np.pi - radians) * diameter / 2)[:np.newaxis]
    start[:, :, 1] = (np.sin(np.pi - radians) * diameter / 2)[:np.newaxis]

    arc_points = np.zeros((np.size(arc_radians, 0), 2))
    arc_points[:, 0] = np.cos(arc_radians) * diameter
    arc_points[:, 1] = np.sin(arc_radians) * diameter

    for i, radian in enumerate(radians):
        end[:, i, 0] = start[:, i, 0] + arc_points[:, 0] * np.cos(radian) + arc_points[:, 1] * np.sin(
            radian)
        end[:, i, 1] = start[:, i, 1] - arc_points[:, 0] * np.sin(radian) + arc_points[:, 1] * np.cos(
            radian)

    if extent:
        return start, end, [radians[0], radians[-1], arc_radians[0], arc_radians[-1]]
    else:
        return start, end


Material = namedtuple('Material', 'color absorbance fission')
Solid = namedtuple('Solid', 'segments in_material out_material')
FlatGeometry = namedtuple('FlatGeometry', 'segments absorbance fission')


def draw(solids, show_normals=False):
    for solid in solids:
        color = solid.in_material.color
        xs = np.ravel(solid.segments[:, :, 0])
        ys = np.ravel(solid.segments[:, :, 1])
        plt.fill(xs, ys, color=color, zorder=1)

        if show_normals:
            normals = normal(solid.segments)
            centers = center(solid.segments)
            for (norm, cent) in zip(normals, centers):
                plt.arrow(cent[0], cent[1], norm[0], norm[1], width=0.01, color=color, zorder=10)


def flatten(solids):
    num_total_segments = 0
    for solid in solids:
        num_total_segments += len(solid.segments)

    flat_geom = FlatGeometry(segments=np.zeros((num_total_segments, 2, 2)),
                             absorbance=np.zeros((num_total_segments, 2)),
                             fission=np.zeros((num_total_segments, 2)))

    index = 0
    for solid in solids:
        solid_slice = slice(index, index + len(solid.segments))
        flat_geom.segments[solid_slice] = solid.segments
        flat_geom.absorbance[solid_slice] = [solid.in_material.absorbance, solid.out_material.absorbance]
        flat_geom.fission[solid_slice] = [solid.in_material.fission, solid.out_material.fission]
        index += len(solid.segments)

    return flat_geom


class Grid(object):
    def __init__(self, width, height, num_x, num_y):
        xs = np.linspace(-width / 2, width / 2, num_x + 1)
        ys = np.linspace(height / 2, -height / 2, num_y + 1)
        self.points = np.zeros((len(ys), len(xs), 2))
        self.points[..., 0], self.points[..., 1] = np.meshgrid(xs, ys)

    @property
    def num_x(self):
        return np.size(self.points, 1) - 1

    @property
    def num_y(self):
        return np.size(self.points, 0) - 1

    @property
    def num_cells(self):
        return self.num_x * self.num_y

    def cell(self, i):
        if i > self.num_cells - 1:
            raise IndexError

        ix = i % (np.size(self.points, 1) - 1)
        iy = i // (np.size(self.points, 1) - 1)

        grid_points = np.zeros((4, 2), dtype=np.double)
        grid_points[0] = self.points[iy, ix]
        grid_points[1] = self.points[iy, ix + 1]
        grid_points[2] = self.points[iy + 1, ix + 1]
        grid_points[3] = self.points[iy + 1, ix]

        return grid_points

    def draw(self):
        for i in range(np.size(self.points, 1)):
            plt.plot(self.points[(0, -1), i, 0], self.points[(0, -1), i, 1], color='black')
        for i in range(np.size(self.points, 0)):
            plt.plot(self.points[i, (0, -1), 0], self.points[i, (0, -1), 1], color='black')
