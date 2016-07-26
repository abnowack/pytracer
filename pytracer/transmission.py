import numpy as np
from . import transmission_c as trans_c
from . import geometry as geo

_intersect_cache = np.empty((100, 2), dtype=np.double)
_index_cache = np.empty(100, dtype=np.int)


def point_segment_distance(px, py, x0, x1, y0, y1):
    length2 = (x1 - x0) ** 2 + (y1 - y0) ** 2
    if length2 <= 0:
        return np.sqrt((px - x0) ** 2 + (px - y0) ** 2)
    t = ((px - x0) * (x1 - x0) + (py - y0) * (y1 - y0)) / length2
    t = min(1, t)
    t = max(0, t)
    projection_x = x0 + t * (x1 - x0)
    projection_y = y0 + t * (y1 - y0)
    distance = np.sqrt((px - projection_x) ** 2 + (py - projection_y) ** 2)
    return distance


def sign_line(x, y, x1, y1, x2, y2):
    return (x - x1) * (y1 - y2) - (y - y1) * (x1 - x2)


def find_absorbance_at_point(point, flat_geom):
    """ Based on looking at segment with smallest distance """
    min_dist_absorbance = None
    min_distance = 1e99
    for i in range(np.size(flat_geom.segments, 0)):
        distance = point_segment_distance(point[0], point[1], flat_geom.segments[i, 0, 0], flat_geom.segments[i, 1, 0],
                                          flat_geom.segments[i, 0, 1], flat_geom.segments[i, 1, 1])
        if distance < min_distance:
            min_distance = distance
            is_outer = sign_line(point[0], point[1], flat_geom.segments[i, 0, 0], flat_geom.segments[i, 0, 1],
                                 flat_geom.segments[i, 1, 0], flat_geom.segments[i, 1, 1])
            if is_outer == 0:
                min_dist_absorbance = (flat_geom.absorbance[i, 1] + flat_geom.absorbance[i, 0]) / 2
            elif is_outer > 0:
                min_dist_absorbance = flat_geom.absorbance[i, 1]
            else:
                min_dist_absorbance = flat_geom.absorbance[i, 0]
    return min_dist_absorbance


def intersections(start, end, segments, intersect_cache=None, index_cache=None, ray=False):
    if intersect_cache is None:
        intersect_cache = _intersect_cache
    if index_cache is None:
        index_cache = _index_cache

    num_intersects = trans_c.intersections(start, end, segments, intersect_cache, index_cache, ray)

    return intersect_cache[:num_intersects], index_cache[:num_intersects]


def absorbance(start, end, segments, seg_absorbance, universe_absorbance=0.0,
               intersect_cache=None, index_cache=None):
    if intersect_cache is None:
        intersect_cache = _intersect_cache
    if index_cache is None:
        index_cache = _index_cache

    return trans_c.absorbance(start, end, segments, seg_absorbance, universe_absorbance,
                              intersect_cache, index_cache)


def attenuation(start, end, segments, seg_absorbance, universe_absorbance=0.0,
                intersect_cache=None, index_cache=None):
    return np.exp(-absorbance(**locals()))


def absorbances(start, end, segments, seg_absorbance, universe_absorbance=0.0,
                intersect_cache=None, index_cache=None, absorbance_cache=None):
    if intersect_cache is None:
        intersect_cache = _intersect_cache
    if index_cache is None:
        index_cache = _index_cache
    if absorbance_cache is None:
        absorbance_cache = np.zeros(len(start), dtype=np.double)

    trans_c.absorbances(start, end, segments, seg_absorbance, universe_absorbance,
                        intersect_cache, index_cache, absorbance_cache)

    return absorbance_cache


def attenuations(start, end, segments, seg_absorbance, universe_absorbance=0.0,
                 intersect_cache=None, index_cache=None, absorbance_cache=None):
    absorb = absorbances(**locals())
    np.exp(-absorb, absorb)
    return absorb


def scan(flat_geom, start, end):
    absorb = np.zeros((start.shape[:-1]), dtype=np.double)
    flat_start = start.reshape(-1, start.shape[-1])
    flat_end = end.reshape(-1, end.shape[-1])

    absorbances(flat_start, flat_end, flat_geom.segments, flat_geom.absorbance, 0, absorbance_cache=absorb.ravel())

    return absorb


def grid_response(flat_geom, grid, start, end):
    unit_m = geo.Material('black', 1, 0)
    vacuum = geo.Material('white', 0, 0)

    flat_start = start.reshape(-1, start.shape[-1])
    flat_end = end.reshape(-1, end.shape[-1])

    response = np.zeros((grid.num_cells,) + start.shape[:-1])
    response_shape = response.shape
    response = response.reshape((response.shape[0], response.shape[1] * response.shape[2]))

    for i in range(grid.num_cells):
        cell_geom = [geo.Solid(geo.convert_points_to_segments(grid.cell(i), circular=True), unit_m, vacuum)]
        cell_flat = geo.flatten(cell_geom)

        absorbances(flat_start, flat_end, cell_flat.segments, cell_flat.absorbance,
                    absorbance_cache=response[i, :])

    return response.reshape(response_shape)
