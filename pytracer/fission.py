import numpy as np
from . import geometry as geo
from . import transmission

_fission_segment_cache = np.empty((100, 2, 2), dtype=np.double)
_fission_value_cache = np.empty(100, dtype=np.int)


def point_is_inner_segment_side(x, y, segments):
    return (x - segments[:, 0, 1]) * (segments[:, 0, 1] - segments[:, 1, 1]) - (y - segments[:, 0, 1]) * (
        segments[:, 0, 0] - segments[:, 1, 0])


def find_fission_segments(start, end, flat_geom, fission_segments=None, fission_values=None):
    if fission_segments is None:
        fission_segments = _fission_segment_cache
    if fission_values is None:
        fission_values = _fission_value_cache

    fission_segment_count = 0
    intersects, indexes = transmission.intersections(start, end, flat_geom.segments)
    if np.size(intersects, 0) == 0:
        return None, None

    # get locations where intersections are on segments containing a fissionable material
    is_fission_boundary = np.where(flat_geom.fission[indexes] > 0)[0]  # checks inner and outer
    f_intersects = intersects[is_fission_boundary]
    f_indexes = indexes[is_fission_boundary]

    if np.size(f_intersects, 0) == 0:
        return None, None

    # sort by distance from start point
    distances = np.sum((f_intersects - start) ** 2, axis=1)
    distance_order = np.argsort(distances)
    f_intersects = f_intersects[distance_order]
    f_indexes = f_indexes[distance_order]

    f_value = flat_geom.fission[f_indexes]
    f_norms = geo.normal(flat_geom.segments[f_indexes])
    f_dot = np.sign(point_is_inner_segment_side(start[0], start[1], flat_geom.segments[f_indexes]))

    # determine if start begins in a fissionable material
    if f_dot[0] > 0 and f_value[0, 0] > 0:
        fission_segments[fission_segment_count] = [start, f_intersects[0]]
        fission_values[fission_segment_count] = f_value[0, 0]
        fission_segment_count += 1
    elif f_dot[0] < 0 and f_value[0, 1] > 0:
        fission_segments[fission_segment_count] = [start, f_intersects[0]]
        fission_values[fission_segment_count] = f_value[0, 1]
        fission_segment_count += 1

    # iterate through pairs of points, determine if they are a fissionable segment
    for i in range(1, np.size(f_indexes) - 1):
        if (f_dot[i] > 0 and f_value[i, 1] > 0) or (f_dot[i] < 0 and f_value[i, 0] > 0):
            if (f_dot[i + 1] > 0 and f_value[i + 1, 0] > 0) or (f_dot[i + 1] < 0 and f_value[i + 1, 1] > 0):
                fission_segments[fission_segment_count] = [f_intersects[i], f_intersects[i + 1]]
                fission_values[fission_segment_count] = f_value[i, 1]
                fission_segment_count += 1

    # determine if end terminates in a fissionable material
    if f_dot[-1] > 0 and f_value[-1, 1] > 0:
        fission_segments[fission_segment_count] = [f_intersects[-1], end]
        fission_values[fission_segment_count] = f_value[-1, 1]
        fission_segment_count += 1
    elif f_dot[-1] < 0 and f_value[-1, 0] > 0:
        fission_segments[fission_segment_count] = [f_intersects[-1], end]
        fission_values[fission_segment_count] = f_value[-1, 0]
        fission_segment_count += 1

    return fission_segments[:fission_segment_count], fission_values[:fission_segment_count]
