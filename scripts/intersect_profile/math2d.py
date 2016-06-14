import numpy as np
from itertools import izip
from math2d_c import intersections as isect


def norm(segment):
    dx = segment[0, 1] - segment[1, 1]
    dy = segment[1, 0] - segment[0, 0]
    length = np.sqrt(dx ** 2. + dy ** 2.)
    return np.array([dx / length, dy / length], dtype=np.float32)


def intersect(segment, other_segment, other_is_ray=False):
    epsilon = 1e-15
    p, q = segment[0], other_segment[0]
    r, s = segment[1] - segment[0], other_segment[1] - other_segment[0]

    denom = r[0] * s[1] - r[1] * s[0]

    # colinear or parallel
    if denom == 0.:
        return None

    u_num = (q - p)[0] * r[1] - (q - p)[1] * r[0]
    t_num = (q - p)[0] * s[1] - (q - p)[1] * s[0]
    t, u = t_num / denom, u_num / denom
    intersection = p + t * r

    # contained with both line segments
    # must shift over line segment by epsilon to prevent double overlapping
    if -epsilon < t < 1. - epsilon:
        if not other_is_ray or 0. < u <= 1.:
            return intersection


def intersections(segments, start, end, ray=False):
    # intersect_segment = np.array([start, end])
    # intercepts, indexes = [], []
    #
    # for i, segment in enumerate(segments):
    #     intercept = intersect(segment, intersect_segment, ray)
    #     if intercept is not None:
    #         intercepts.append(np.array(intercept))
    #         indexes.append(i)
    #
    # return intercepts, indexes

    return isect(segments, np.array([start, end]), ray)


def attenuation_length(segments, start, end, inner_attenuation, outer_attenuation, universe_attenuation):
    intercepts, indexes = intersections(segments, start, end)
    no_segment_intercepts = False

    # If no intersection must determine what material we are within by tracing a ray
    if len(intercepts) == 0:
        intercepts, indexes = intersections(segments, start, end, ray=True)
        no_segment_intercepts = True
    # No intersection through a ray, must be outside the object, return atten_length from universe material
    if len(intercepts) == 0:
        return np.linalg.norm(start - end) * universe_attenuation

    distances = np.linalg.norm(np.add(intercepts, -start), axis=1)
    distances_argmin = np.argmin(distances)
    closest_index = indexes[distances_argmin]
    closest_intercept = intercepts[distances_argmin]
    closest_normal = norm(segments[closest_index])
    start_sign = np.sign(np.dot(start - closest_intercept, closest_normal))

    if start_sign > 0:
        outer_atten = outer_attenuation[closest_index]
        atten_length = np.linalg.norm(start - end) * outer_atten
    else:
        inner_atten = inner_attenuation[closest_index]
        atten_length = np.linalg.norm(start - end) * inner_atten

    # No segment intercept, so return the beginning to end atten_length
    if no_segment_intercepts:
        return atten_length

    # Had intersections, so add up all individual atten_lengths between start to end
    for intercept, index in izip(intercepts, indexes):
        normal = norm(segments[index])
        start_sign = np.sign(np.dot(start - intercept, normal))
        inner_atten = inner_attenuation[index]
        outer_atten = outer_attenuation[index]

        atten_length += start_sign * np.linalg.norm(intercept - end) * (inner_atten - outer_atten)

    return atten_length
