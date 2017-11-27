import numpy as np
from . import geometry as geo
from . import transmission
from . import fission_c
from . import transmission_c
from . import neutron_chain as chain

_fission_segment_cache = np.empty((100, 2, 2), dtype=np.double)
_fission_value_cache = np.empty(100, dtype=np.double)
_array1D_cache = np.empty(100, dtype=np.double)


### CYTHON TODO BEGIN
def distance(x1, y1, x2, y2):
    tmp = 0

    tmp = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
    return np.sqrt(tmp)


def sign_line(x, y, x1, y1, x2, y2):
    return (x - x1) * (y1 - y2) + (y - y1) * (x2 - x1)


def point_segment_distance(px, py, x0, x1, y0, y1):
    length_sq = (x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0)
    if length_sq <= 0:
        distance = np.sqrt((px - x0) * (px - x0) + (py - y0) * (py - y0))
        return distance

    t = (px - x0) * (x1 - x0) + (py - y0) * (y1 - y0)
    t /= length_sq
    if t > 1:
        t = 1
    elif t < 0:
        t = 0

    projection_x = x0 + t * (x1 - x0)
    projection_y = y0 + t * (y1 - y0)
    distance = (px - projection_x) * (px - projection_x) + (py - projection_y) * (py - projection_y)
    distance = np.sqrt(distance)
    return distance


def p_at_point(point_x, point_y, segments, pfuncrefs, pfuncs):
    """ Based on looking at segment with smallest distance """
    min_distance = 1e99
    point_absorbance = 0

    for i in range(segments.shape[0]):
        distance = point_segment_distance(point_x, point_y, segments[i, 0, 0], segments[i, 1, 0],
                                          segments[i, 0, 1], segments[i, 1, 1])
        if distance < min_distance:
            min_distance = distance
            is_outer = sign_line(point_x, point_y, segments[i, 0, 0], segments[i, 0, 1],
                                 segments[i, 1, 0], segments[i, 1, 1])
            in_p = 0
            out_p = 0
            if pfuncrefs[i, 0] > 0:
                in_p = pfuncs[pfuncrefs[i, 0]](point_x, point_y)
            if pfuncrefs[i, 1] > 0:
                out_p = pfuncs[pfuncrefs[i, 1]](point_x, point_y)

            if is_outer == 0:
                point_absorbance = (in_p + out_p) / 2
            elif is_outer > 0:
                point_absorbance = out_p
            else:
                point_absorbance = in_p
    return point_absorbance


### CYTHON TODO END




def fissionval_at_point(point, flat_geom):
    return transmission_c.absorbance_at_point(point[0], point[1], flat_geom.segments, flat_geom.fission)


def fissionval_image(xs, ys, flat_geom):
    image = np.zeros((np.size(xs, 0), np.size(ys, 0)), dtype=np.double)
    extent = [xs[0], xs[-1], ys[0], ys[-1]]

    transmission_c.absorbance_image(image, xs, ys, flat_geom.segments, flat_geom.fission)

    return image, extent


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
            p_value = p_at_point(_array1D_cache[0], _array1D_cache[1], flat_geom.segments,
                                 flat_geom.pfuncrefs, flat_geom.pfuncs)

            nudist_interp = chain.interpolate_p(matrix, p_value, p_range, method='linear', log_interpolate=False)
            nudist_arr = nudist_interp[0]

            #
            # print(nudist_interp.shape)
            # print(transmission._index_cache.shape)

            prob += fission_c.probability_point_neutron(flat_geom.segments, flat_geom.absorbance, _array1D_cache,
                                                        detector_segments,
                                                        0.0, start, k, nudist_arr, fission_value,
                                                        transmission._intersect_cache,
                                                        transmission._index_cache)

        prob *= segment_length / num_segment_points
    return prob


# TODO Cython
def scan(source, neutron_paths, detector_points, flat_geom, k, matrix, p_range):
    probs = np.zeros((np.size(source, 0), np.size(neutron_paths, 0)), dtype=np.double)
    print(np.size(source, 0) * np.size(neutron_paths, 0))
    for i in range(np.size(source, 0)):
        detector_segments = geo.convert_points_to_segments(detector_points[:, i])
        for j in range(np.size(neutron_paths, 0)):
            probs[i, j] = probability_path_neutron(source[i], neutron_paths[j, i], flat_geom, detector_segments,
                                                   k, matrix, p_range)

    return probs


# TODO Cython
def grid_cell_response(source, neutron_path, detector_segments, flat_cell_geom, flat_geom, k, nu_dist,
                       num_segment_points=5):
    """ Return array of size grid containing single fission response"""
    segment, val = find_fission_segments(source, neutron_path, flat_cell_geom)

    if len(segment) == 0:
        return 0

    segment = segment[0]
    prob = probability_segment_neutron_grid_c(source, segment, flat_geom, detector_segments, k, nu_dist)

    return prob


# TODO Cython
def grid_response_scan(source, neutron_paths, detector_points, flat_cell_geom, flat_geom, k, nu_dist):
    probs = np.zeros((np.size(source, 0), np.size(neutron_paths, 0)), dtype=np.double)
    for i in range(np.size(source, 0)):
        detector_segments = geo.convert_points_to_segments(detector_points[:, i])
        for j in range(np.size(neutron_paths, 0)):
            probs[i, j] = grid_cell_response(source[i], neutron_paths[j, i], detector_segments, flat_cell_geom,
                                             flat_geom, k, nu_dist)

    return probs


# TODO Cython
def grid_response(source, neutron_paths, detector_points, grid, flat_geom, k, nu_dist):
    unit_m = geo.Material('black', 1, 1)
    vacuum = geo.Material('white', 0, 0)

    response = np.zeros((grid.num_cells, np.size(source, 0), np.size(neutron_paths, 0)), dtype=np.double)

    for i in range(grid.num_cells):
        print(i, ' / ', grid.num_cells)
        cell_geom = [geo.Solid(geo.convert_points_to_segments(grid.cell(i), circular=True), unit_m, vacuum)]
        cell_flat = geo.flatten(cell_geom)

        response[i, :, :] = grid_response_scan(source, neutron_paths, detector_points, cell_flat, flat_geom,
                                               k, nu_dist)

    return response
