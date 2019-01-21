import numpy as np
from . import transmission_c as trans_c
from . import geometry as geo

_intersect_cache = np.empty((100, 2), dtype=np.double)
_index_cache = np.empty(100, dtype=np.int)


def absorbance_at_point(point, flat_geom):
    return trans_c.absorbance_at_point(point[0], point[1], flat_geom.segments, flat_geom.absorbance)


def absorbance_image(xs, ys, flat_geom):
    image = np.zeros((np.size(xs, 0), np.size(ys, 0)), dtype=np.double)
    extent = [xs[0], xs[-1], ys[0], ys[-1]]

    trans_c.absorbance_image(image, xs, ys, flat_geom.segments, flat_geom.absorbance)

    return image.T, extent


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


def grid_response(flat_geom, grid, start, end):
    unit_m = geo.Material('black', 1, 0, 0)
    vacuum = geo.Material('white', 0, 0, 0)

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
