import numpy as np
from . import geometry as geo
from . import transmission

_fission_segment_cache = np.empty((100, 2, 2), dtype=np.double)
_fission_value_cache = np.empty(100, dtype=np.double)


def point_is_outer_segment_side(x, y, segments):
    return (x - segments[:, 0, 0]) * (segments[:, 0, 1] - segments[:, 1, 1]) - (y - segments[:, 0, 1]) * (
        segments[:, 0, 0] - segments[:, 1, 0]) > 0


def find_fission_segments(start, end, flat_geom, fission_segments=None, fission_values=None):
    if fission_segments is None:
        fission_segments = _fission_segment_cache
    if fission_values is None:
        fission_values = _fission_value_cache

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
    if start_on_outer_side[0] and value[0, 0] > 0:
        fission_segments[segment_count] = [start, intersects[0]]
        fission_values[segment_count] = value[0, 0]
        segment_count += 1
    elif not start_on_outer_side[0] and value[0, 1] > 0:
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
        elif not start_on_outer_side[i] and value[i, 0] > 0:
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


def break_segment(segment, num_points=5):
    points = np.zeros((num_points, 2), dtype=segment.dtype)
    points[:, 0] = np.linspace(segment[0, 0], segment[1, 0], num_points + 2)
    points[:, 1] = np.linspace(segment[0, 1], segment[1, 1], num_points + 2)
    return points


def probability_detect(fission_position, flat_geom, detector_segments):
    # calc attenuation from fission position to center of detector segments
    # could calc the average over a segment if attenuation varies enough over the segment
    end_positions = geo.center(detector_segments)
    start_positions = np.tile(fission_position, (len(end_positions), 1))
    out_attenuations = transmission.attenuations(start_positions, end_positions, flat_geom.segments,
                                                 flat_geom.attenuation)
    exit_prob = np.exp(-out_attenuations)

    # calc solid angle of detector from fission_point
    solid_angles = geo.solid_angle(detector_segments, fission_position)
    prob_solid_angle = solid_angles / (2 * np.pi)

    prob_detect = np.sum(prob_solid_angle * exit_prob)

    return prob_detect


def probability_in(source, fission_position, flat_geom):
    prob_atten = transmission.attenuation(source, fission_position, flat_geom.segments, flat_geom.attenuation)
    return np.exp(-prob_atten)


def probability_out_single(fission_position, flat_geom, detector_segments, avg_nu):
    prob_detect = probability_detect(fission_position, flat_geom, detector_segments)
    prob_out_single = np.exp(-prob_detect * avg_nu) * prob_detect * avg_nu
    return prob_out_single


def probability_per_ds_neutron_single(source, fission_position, flat_geom, mu_fission, detector_segments, avg_nu):
    prob_in = probability_in(source, fission_position, flat_geom)
    prob_out = probability_out_single(fission_position, flat_geom, detector_segments, avg_nu)
    prob_per_ds_fission = mu_fission

    return prob_in * prob_per_ds_fission * prob_out


def probability_segment_neutron_single(source, fission_segment, mu_fission, flat_geom, detector_segments, avg_nu,
                                       num_segment_points=5):
    fission_positions = np.zeros((num_segment_points, 2), dtype=np.double)
    epsilon = 1.0e-6
    x_start, x_end = epsilon * fission_segment[1, 0] + (1 - epsilon) * fission_segment[0, 0], \
                     (1 - epsilon) * fission_segment[1, 0] + epsilon * fission_segment[0, 0],
    y_start, y_end = epsilon * fission_segment[1, 1] + (1 - epsilon) * fission_segment[0, 1], \
                     (1 - epsilon) * fission_segment[1, 1] + epsilon * fission_segment[0, 1],

    fission_positions[:, 0] = np.linspace(x_start, x_end, num_segment_points)
    fission_positions[:, 1] = np.linspace(y_start, y_end, num_segment_points)

    segment_length = np.sqrt(
        (fission_segment[0, 0] - fission_segment[1, 0]) ** 2 + (fission_segment[0, 1] - fission_segment[1, 1]))

    segment_probability = 0.

    for fission_position in fission_positions:
        prob_ds = probability_per_ds_neutron_single(source, fission_position, flat_geom, mu_fission, detector_segments,
                                                    avg_nu)
        segment_probability += prob_ds

    segment_probability *= segment_length

    return segment_probability


def probability_path_neutron_single(start, end, flat_geom, detector_segments, avg_nu):
    segments, values = find_fission_segments(start, end, flat_geom)

    prob = 0
    for (segment, value) in zip(segments, values):
        prob += probability_segment_neutron_single(start, segment, value, flat_geom, detector_segments, avg_nu)

    return prob


def probability_out_double(fission_position, flat_geom, detector_segments, avg_nu):
    prob_detect = probability_detect(fission_position, flat_geom, detector_segments)
    prob_out_double = 0.5 * np.exp(-prob_detect * avg_nu) * (prob_detect * avg_nu) ** 2
    return prob_out_double


def probability_per_ds_neutron_double(source, fission_position, flat_geom, mu_fission, detector_segments, avg_nu):
    prob_in = probability_in(source, fission_position, flat_geom)
    prob_out = probability_out_double(fission_position, flat_geom, detector_segments, avg_nu)
    prob_per_ds_fission = mu_fission

    return prob_in * prob_per_ds_fission * prob_out


def probability_segment_neutron_double(source, fission_segment, mu_fission, flat_geom, detector_segments, avg_nu,
                                       num_segment_points=5):
    fission_positions = np.zeros((num_segment_points, 2), dtype=np.double)
    epsilon = 1.0e-6
    x_start, x_end = epsilon * fission_segment[1, 0] + (1 - epsilon) * fission_segment[0, 0], \
                     (1 - epsilon) * fission_segment[1, 0] + epsilon * fission_segment[0, 0],
    y_start, y_end = epsilon * fission_segment[1, 1] + (1 - epsilon) * fission_segment[0, 1], \
                     (1 - epsilon) * fission_segment[1, 1] + epsilon * fission_segment[0, 1],

    fission_positions[:, 0] = np.linspace(x_start, x_end, num_segment_points)
    fission_positions[:, 1] = np.linspace(y_start, y_end, num_segment_points)

    segment_length = np.sqrt(
        (fission_segment[0, 0] - fission_segment[1, 0]) ** 2 + (fission_segment[0, 1] - fission_segment[1, 1]))

    segment_probability = 0.

    for fission_position in fission_positions:
        prob_ds = probability_per_ds_neutron_double(source, fission_position, flat_geom, mu_fission, detector_segments,
                                                    avg_nu)
        segment_probability += prob_ds

    segment_probability *= segment_length

    return segment_probability


def probability_path_neutron_double(start, end, flat_geom, detector_segments, avg_nu):
    segments, values = find_fission_segments(start, end, flat_geom)

    prob = 0
    for (segment, value) in zip(segments, values):
        prob += probability_segment_neutron_double(start, segment, value, flat_geom, detector_segments, avg_nu)

    return prob


def grid_response_single(start, end, grid, flat_geom, detector_segments, avg_nu):
    """ Return array of size grid containing single fission response"""

    unit_m = geo.Material('black', 1, 1)
    vacuum = geo.Material('white', 0, 0)

    flat_start = start.reshape(-1, start.shape[-1])
    flat_end = end.reshape(-1, end.shape[-1])

    response = np.zeros((grid.num_cells,) + start.shape[:-1])
    response_shape = response.shape
    response = response.reshape((response.shape[0], response.shape[1] * response.shape[2]))

    for i in range(grid.num_cells):
        cell_geom = [geo.Solid(geo.convert_points_to_segments(grid.cell(i)), unit_m, vacuum)]
        cell_flat = geo.flatten(cell_geom)

        for j, (s, e) in enumerate(zip(start, end)):
            segments, values = find_fission_segments(s, e, cell_flat)
            response[i, j] = probability_segment_neutron_single(s, segments[0], values[0], flat_geom, detector_segments,
                                                                avg_nu)

    return response.reshape(response_shape)
