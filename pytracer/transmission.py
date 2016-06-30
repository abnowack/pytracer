import numpy as np
from . import transmission_c as trans_c
from . import geometry as geo

_intersect_cache = np.empty((100, 2), dtype=np.double)
_index_cache = np.empty(100, dtype=np.int)


def intersections(start, end, segments, intersect_cache=None, index_cache=None, ray=False):
    if intersect_cache is None:
        intersect_cache = _intersect_cache
    if index_cache is None:
        index_cache = _index_cache

    num_intersects = trans_c.intersections(start, end, segments, intersect_cache, index_cache, ray)

    return intersect_cache[:num_intersects], index_cache[:num_intersects]


def attenuation(start, end, segments, seg_attenuation, universe_attenuation=0.0,
                intersect_cache=None, index_cache=None):
    if intersect_cache is None:
        intersect_cache = _intersect_cache
    if index_cache is None:
        index_cache = _index_cache

    return trans_c.attenuation(start, end, segments, seg_attenuation, universe_attenuation,
                               intersect_cache, index_cache)


def attenuations(start, end, segments, seg_attenuation, universe_attenuation=0.0,
                 intersect_cache=None, index_cache=None, attenuation_cache=None):
    if intersect_cache is None:
        intersect_cache = _intersect_cache
    if index_cache is None:
        index_cache = _index_cache
    if attenuation_cache is None:
        attenuation_cache = np.zeros(len(start), dtype=np.double)

    trans_c.attenuations(start, end, segments, seg_attenuation, universe_attenuation,
                         intersect_cache, index_cache, attenuation_cache)

    return attenuation_cache


def scan(flat_geom, start, end):
    atten = np.zeros((start.shape[:-1]), dtype=np.double)
    flat_start = start.reshape(-1, start.shape[-1])
    flat_end = end.reshape(-1, end.shape[-1])

    attenuations(flat_start, flat_end, flat_geom.segments, flat_geom.attenuation, 0, attenuation_cache=atten.ravel())

    return atten


def grid_response(flat_geom, grid, start, end):
    unit_m = geo.Material('black', 1, 0)
    vacuum = geo.Material('white', 0, 0)

    flat_start = start.reshape(-1, start.shape[-1])
    flat_end = end.reshape(-1, end.shape[-1])

    response = np.zeros((grid.num_cells,) + start.shape[:-1])
    response_shape = response.shape
    response = response.reshape((response.shape[0], response.shape[1] * response.shape[2]))

    for i in range(grid.num_cells):
        cell_geom = [geo.Solid(geo.convert_points_to_segments(grid.cell(i)), unit_m, vacuum)]
        cell_flat = geo.flatten(cell_geom)

        attenuations(flat_start, flat_end, cell_flat.segments, cell_flat.attenuation,
                     attenuation_cache=response[i, :])

    return response.reshape(response_shape)
