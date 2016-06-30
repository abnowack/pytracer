import numpy as np
from . import geometry as geo

def propagate_fission_ray(sim, start, end, n):
    segments, macro_fissions = sim.geometry.fission_segments(start, end)
    segment_probs = []
    for segment in segments:
        single_fission_prob = propagate_fissions_segment(sim, segment, n)
        segment_probs.append(single_fission_prob)
    total_fission_prob = np.sum(segment_probs, axis=0)
    return total_fission_prob


def propagate_fissions_segment(sim, segment, n=5):
    point_0, point_1 = segment[0], segment[1]
    # generate points along fission segment
    # use trapezoid rule on uniform spacing
    # int [f(x = [a, b]) dx]  ~= (b - a) / (2 * N) [ f(a) + f(b) +  ]
    points = [point_0 + (point_1 - point_0) * t for t in np.linspace(0.01, 0.99, n)]  # TODO : error if t = 1
    values = np.zeros((len(points), len(sim.detector.segments)))
    integral = np.zeros((len(sim.detector.segments)))
    for i, point in enumerate(points):
        values[i, :] = propagate_fissions_point_detector(sim, point)
    integral[:] = np.linalg.norm(point_1 - point_0) / (n - 1) * (
        values[0, :] + 2 * np.sum(values[1:-1, :], axis=0) + values[-1, :])
    return integral


def propagate_fissions_point_detector(sim, point):
    """
    Calculate probability of induced fission being detected over detector plane.

    nu = 1 for now, not using macro_fission
    """
    detector_solid_angle = math2d.solid_angle(sim.detector.segments, point) / (2 * np.pi)  # returns 200,200
    in_attenuation_length = sim.geometry.attenuation_length(sim.source.pos, point)
    segment_centers = math2d.center(sim.detector.segments)
    out_attenuation_lengths = np.array([sim.geometry.attenuation_length(point, center) for center in segment_centers])

    prob = np.exp(-in_attenuation_length) * np.multiply(detector_solid_angle, np.exp(-out_attenuation_lengths))

    return prob


def calc_fission_prob(sim, rays, r=50):
    fission_probs = np.zeros((np.size(rays, 0), sim.detector.nbins))

    for i, ray in enumerate(rays):
        end = sim.source.pos + 50 * ray

        fission_probs[i, :] = propagate_fission_ray(sim, sim.source.pos, end, n=5)

