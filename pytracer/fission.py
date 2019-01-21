import numpy as np
from . import geometry as geo
from . import transmission
from . import fission_c
from . import transmission_c
from . import neutron_chain as chain

_fission_segment_cache = np.empty((100, 2, 2), dtype=np.double)
_fission_value_cache = np.empty(100, dtype=np.double)
_array1D_cache = np.empty(100, dtype=np.double)


def fissionval_at_point(point, flat_geom):
    return transmission_c.absorbance_at_point(point[0], point[1], flat_geom.segments, flat_geom.fission)


def fissionval_image(xs, ys, flat_geom):
    image = np.zeros((np.size(xs, 0), np.size(ys, 0)), dtype=np.double)
    extent = [xs[0], xs[-1], ys[0], ys[-1]]

    transmission_c.absorbance_image(image, xs, ys, flat_geom.segments, flat_geom.fission)

    return image.T, extent


# TODO Cython
def find_fission_segments(start, end, flat_geom, fission_segments=None, fission_values=None):
    if fission_segments is None:
        fission_segments = _fission_segment_cache
    if fission_values is None:
        fission_values = _fission_value_cache

    def point_is_outer_segment_side(x, y, segments):
        return (x - segments[:, 0, 0]) * (segments[:, 0, 1] - segments[:, 1, 1]) + (y - segments[:, 0, 1]) * (
            segments[:, 1, 0] - segments[:, 0, 0]) > 0

    segment_count = 0
    intersects, indexes = transmission.intersections(np.copy(start), np.copy(end), flat_geom.segments)
    if np.size(intersects, 0) == 0:
        return fission_segments[:0], fission_values[:0]

    # sort intercepts by distance
    distances = np.sum((intersects - start) ** 2, axis=1)
    distance_order = np.argsort(distances)
    intersects = intersects[distance_order]
    indexes = indexes[distance_order]
    value = flat_geom.fission[indexes]
    start_on_outer_side = point_is_outer_segment_side(start[0], start[1], flat_geom.segments[indexes])

    # test if [start, intersect[0]] is fissionable path
    if start_on_outer_side[0] and value[0, 1] > 0:
        fission_segments[segment_count] = [start, intersects[0]]
        fission_values[segment_count] = value[0, 0]
        segment_count += 1
    elif not start_on_outer_side[0] and value[0, 0] > 0:
        fission_segments[segment_count] = [start, intersects[0]]
        fission_values[segment_count] = value[0, 1]
        segment_count += 1

    # test all intervening segments
    for i in range(np.size(intersects, 0) - 1):
        if start_on_outer_side[i] and value[i, 0] > 0:
            if start_on_outer_side[i + 1] and value[i + 1, 1] > 0:
                fission_segments[segment_count] = [intersects[i], intersects[i + 1]]
                fission_values[segment_count] = value[i + 1, 1]
                segment_count += 1
            elif not start_on_outer_side[i + 1] and value[i + 1, 0] > 0:
                fission_segments[segment_count] = [intersects[i], intersects[i + 1]]
                fission_values[segment_count] = value[i + 1, 0]
                segment_count += 1
        elif not start_on_outer_side[i] and value[i, 1] > 0:
            if start_on_outer_side[i + 1] and value[i + 1, 1] > 0:
                fission_segments[segment_count] = [intersects[i], intersects[i + 1]]
                fission_values[segment_count] = value[i + 1, 1]
                segment_count += 1
            elif not start_on_outer_side[i + 1] and value[i + 1, 0] > 0:
                fission_segments[segment_count] = [intersects[i], intersects[i + 1]]
                fission_values[segment_count] = value[i + 1, 0]
                segment_count += 1

    # test if [intersect[-1], end] is fissionable path
    if start_on_outer_side[-1] and value[-1, 0] > 0:
        fission_segments[segment_count] = [intersects[-1], end]
        fission_values[segment_count] = value[-1, 0]
        segment_count += 1
    elif not start_on_outer_side[-1] and value[-1, 1] > 0:
        fission_segments[segment_count] = [intersects[-1], end]
        fission_values[segment_count] = value[-1, 1]
        segment_count += 1

    return fission_segments[:segment_count], fission_values[:segment_count]


def probability_segment_neutron_c(source, fission_segment, mu_fission, flat_geom, detector_segments, k, nu_dist,
                                  num_segment_points=5):
    return fission_c.probability_segment_neutron(flat_geom.segments, flat_geom.absorbance, fission_segment,
                                                 detector_segments,
                                                 0.0, num_segment_points, source, k, nu_dist, mu_fission,
                                                 transmission._intersect_cache,
                                                 transmission._index_cache, _array1D_cache)


def probability_segment_neutron_grid_c(source, fission_segment, flat_geom, detector_segments, k, nu_dist,
                                       num_segment_points=5):
    return fission_c.probability_segment_neutron_grid(flat_geom.segments, flat_geom.absorbance, fission_segment,
                                                      detector_segments,
                                                      0.0, num_segment_points, source, k, nu_dist,
                                                      transmission._intersect_cache,
                                                      transmission._index_cache, _array1D_cache)


def probability_path_neutron(start, end, flat_geom, detector_segments, k, matrix, p_range):
    num_segment_points = 5
    point = _array1D_cache[:2]

    nudist_arr = np.zeros((21), dtype=np.double)

    fission_segments, fission_values = find_fission_segments(start, end, flat_geom)

    prob = 0
    for (fission_segment, fission_value) in zip(fission_segments, fission_values):
        segment_length = (fission_segment[0, 0] - fission_segment[1, 0]) * \
                         (fission_segment[0, 0] - fission_segment[1, 0])
        segment_length += (fission_segment[0, 1] - fission_segment[1, 1]) * \
                          (fission_segment[0, 1] - fission_segment[1, 1])
        segment_length = np.sqrt(segment_length)

        for i in range(1, num_segment_points + 1):
            _array1D_cache[0] = fission_segment[0, 0] + \
                                (i - 0.5) * (fission_segment[1, 0] - fission_segment[0, 0]) / num_segment_points
            _array1D_cache[1] = fission_segment[0, 1] + \
                                (i - 0.5) * (fission_segment[1, 1] - fission_segment[0, 1]) / num_segment_points

            # calc p at point
            p_value = chain.p_at_point(_array1D_cache, flat_geom)
            # p_value = p_at_point(_array1D_cache[0], _array1D_cache[1], flat_geom.segments,
            #                      flat_geom.pfuncrefs, flat_geom.pfuncs)

            nudist_interp = chain.interpolate_p(matrix, p_value, p_range, method='linear', log_interpolate=False)
            # nudist_arr = nudist_interp[0]
            nudist_arr = nudist_interp

            prob += fission_c.probability_point_neutron(flat_geom.segments, flat_geom.absorbance, _array1D_cache,
                                                        detector_segments,
                                                        0.0, start, k, nudist_arr, fission_value,
                                                        transmission._intersect_cache,
                                                        transmission._index_cache)

        prob *= segment_length / num_segment_points
    return prob


