import numpy as np
from . import geometry as geo
from . import transmission

_fission_segment_cache = np.empty((100, 2, 2), dtype=np.double)
_fission_value_cache = np.empty(100, dtype=np.double)


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


def find_fissionval_at_point(point, flat_geom):
    """ Based on looking at segment with smallest distance """
    min_dist_fission = None
    min_distance = 1e99
    for i in range(np.size(flat_geom.segments, 0)):
        distance = point_segment_distance(point[0], point[1], flat_geom.segments[i, 0, 0], flat_geom.segments[i, 1, 0],
                                          flat_geom.segments[i, 0, 1], flat_geom.segments[i, 1, 1])
        if distance < min_distance:
            min_distance = distance
            is_outer = sign_line(point[0], point[1], flat_geom.segments[i, 0, 0], flat_geom.segments[i, 0, 1],
                                 flat_geom.segments[i, 1, 0], flat_geom.segments[i, 1, 1])
            if is_outer == 0:
                min_dist_fission = (flat_geom.fission[i, 1] + flat_geom.fission[i, 0]) / 2
            elif is_outer > 0:
                min_dist_fission = flat_geom.fission[i, 1]
            else:
                min_dist_fission = flat_geom.fission[i, 0]
    return min_dist_fission


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


def probability_detect(position, flat_geom, detector_segments):
    # calc attenuation from fission position to center of detector segments
    # could calc the average over a segment if attenuation varies enough over the segment
    end_positions = geo.center(detector_segments)
    start_positions = np.tile(position, (len(end_positions), 1))
    out_attenuations = transmission.attenuations(start_positions, end_positions, flat_geom.segments,
                                                 flat_geom.absorbance)
    exit_prob = out_attenuations

    # calc solid angle of detector from fission_point
    solid_angles = geo.solid_angle(detector_segments, position)
    prob_solid_angle = solid_angles / (2 * np.pi)

    prob_detect = np.sum(prob_solid_angle * exit_prob)

    return prob_detect


def probability_in(source, position, flat_geom):
    prob_atten = transmission.attenuation(source, position, flat_geom.segments, flat_geom.absorbance)
    return prob_atten


def probability_out(position, flat_geom, detector_segments, k, avg_nu):
    prob_detect = probability_detect(position, flat_geom, detector_segments)
    if k == 1:
        prob_out = np.exp(-prob_detect * avg_nu) * prob_detect * avg_nu
    elif k == 2:
        prob_out = 0.5 * np.exp(-prob_detect * avg_nu) * (prob_detect * avg_nu) ** 2
    else:
        raise NotImplemented
    return prob_out


def probability_per_ds_neutron(source, position, flat_geom, mu_fission, detector_segments, k, avg_nu):
    prob_in = probability_in(source, position, flat_geom)
    prob_out = probability_out(position, flat_geom, detector_segments, k, avg_nu)
    prob_per_ds_fission = mu_fission

    return prob_in * prob_per_ds_fission * prob_out


def probability_segment_neutron(source, fission_segment, mu_fission, flat_geom, detector_segments, k, avg_nu,
                                num_segment_points=5):
    segment_length = np.sqrt(
        (fission_segment[0, 0] - fission_segment[1, 0]) ** 2 + (fission_segment[0, 1] - fission_segment[1, 1]))

    segment_probability = 0.

    for i in range(1, num_segment_points + 1):
        xi = fission_segment[0, 0] + (i - 0.5) * (fission_segment[1, 0] - fission_segment[0, 0]) / num_segment_points
        yi = fission_segment[0, 1] + (i - 0.5) * (fission_segment[1, 1] - fission_segment[0, 1]) / num_segment_points
        prob_ds = probability_per_ds_neutron(source, np.array([xi, yi]), flat_geom, mu_fission, detector_segments, k,
                                             avg_nu)
        segment_probability += prob_ds * segment_length / num_segment_points

    return segment_probability


def probability_path_neutron(start, end, flat_geom, detector_segments, k, avg_nu):
    segments, values = find_fission_segments(start, end, flat_geom)

    prob = 0
    for (segment, value) in zip(segments, values):
        prob += probability_segment_neutron(start, segment, value, flat_geom, detector_segments, k, avg_nu)

    return prob


def scan(source, neutron_paths, detector_points, flat_geom, k, avg_nu):
    probs = np.zeros((np.size(source, 0), np.size(neutron_paths, 0)), dtype=np.double)
    for i in range(np.size(source, 0)):
        detector_segments = geo.convert_points_to_segments(detector_points[:, i])
        for j in range(np.size(neutron_paths, 0)):
            probs[i, j] = probability_path_neutron(source[i], neutron_paths[j, i], flat_geom, detector_segments,
                                                   k, avg_nu)

    return probs


def grid_cell_response(source, neutron_path, detector_segments, flat_cell_geom, flat_geom, k, avg_nu,
                       num_segment_points=5):
    """ Return array of size grid containing single fission response"""
    segment, val = find_fission_segments(source, neutron_path, flat_cell_geom)

    if len(segment) == 0:
        return 0

    segment = segment[0]

    prob = 0
    length = np.sqrt((segment[1, 0] - segment[0, 0]) ** 2 + (segment[1, 1] - segment[0, 1]) ** 2)

    for i in range(1, num_segment_points + 1):
        xi = segment[0, 0] + (i - 0.5) * (segment[1, 0] - segment[0, 0]) / num_segment_points
        yi = segment[0, 1] + (i - 0.5) * (segment[1, 1] - segment[0, 1]) / num_segment_points
        absorbance = transmission.find_absorbance_at_point(np.array([xi, yi]), flat_geom)
        prob_ds = probability_per_ds_neutron(source, np.array([xi, yi]), flat_geom, absorbance, detector_segments, k,
                                             avg_nu)
        prob += prob_ds * length / num_segment_points

    return prob


def grid_response_scan(source, neutron_paths, detector_points, flat_cell_geom, flat_geom, k, avg_nu):
    probs = np.zeros((np.size(source, 0), np.size(neutron_paths, 0)), dtype=np.double)
    for i in range(np.size(source, 0)):
        detector_segments = geo.convert_points_to_segments(detector_points[:, i])
        for j in range(np.size(neutron_paths, 0)):
            probs[i, j] = grid_cell_response(source[i], neutron_paths[j, i], detector_segments, flat_cell_geom,
                                             flat_geom, k, avg_nu)

    return probs


def grid_response(source, neutron_paths, detector_points, grid, flat_geom, k, avg_nu):
    unit_m = geo.Material('black', 1, 1)
    vacuum = geo.Material('white', 0, 0)

    response = np.zeros((grid.num_cells, np.size(source, 0), np.size(neutron_paths, 0)), dtype=np.double)

    for i in range(grid.num_cells):
        print(i, ' / ', grid.num_cells)
        cell_geom = [geo.Solid(geo.convert_points_to_segments(grid.cell(i), circular=True), unit_m, vacuum)]
        cell_flat = geo.flatten(cell_geom)

        response[i, :, :] = grid_response_scan(source, neutron_paths, detector_points, cell_flat, flat_geom,
                                               k, avg_nu)

    return response
