from mesh import create_hollow, create_rectangle, create_circle
from material import Material
from solid import Solid
from simulation import Simulation
from raytrace import build_shielded_geometry
import math2d

import numpy as np
import matplotlib.pyplot as plt
import sys


def ray_trace_test_geometry():
    air = Material(0.0, color='white')
    steel = Material(0.2, color='red')

    box = create_hollow(create_rectangle(12., 12.), create_rectangle(10., 10.))
    ring = create_hollow(create_circle(12.), create_circle(10.))
    box.rotate(45.)

    sim = Simulation(air, diameter=50.)
    sim.detector.width = 30.
    sim.geometry.solids.append(Solid(ring, steel, air))
    sim.geometry.flatten()

    return sim


def propagate_fission_ray(sim, start, end, n):
    segments, macro_fissions = sim.fission_segments(start, end)
    segment_probs = []
    for i in xrange(len(segments)):
        single_fission_prob = propagate_fissions_segment(sim, segments[i], n)
        segment_probs.append(single_fission_prob)
    total_fission_prob = np.sum(segment_probs, axis=0)
    return total_fission_prob


def propagate_fissions_segment(sim, segment, n=5):
    point_0, point_1 = segment[0], segment[1]
    # generate points along fission segment
    # use trapezoid rule on uniform spacing
    # int [f(x = [a, b]) dx]  ~= (b - a) / (2 * N) [ f(a) + f(b) +  ] 
    points = [point_0 + (point_1 - point_0) * t for t in np.linspace(0.01, 0.99, n)] # TODO : error if t = 1
    values = np.zeros((len(points), len(sim.detector.segments)))
    integral = np.zeros((len(sim.detector.segments)))
    for i in xrange(len(points)):
        values[i, :] = propagate_fissions_point_detector(sim, points[i])
    integral[:] = np.linalg.norm(point_1 - point_0) / (n - 1) * (values[0, :] + 2. * np.sum(values[1:-1, :], axis=0) + values[-1, :])
    return integral 


def propagate_fissions_point_detector(sim, point):
    """
    Calculate probability of induced fission being detected over detector plane.

    nu = 1 for now, not using macro_fission
    """
    detector_solid_angle = math2d.solid_angle(sim.detector.segments, point) / (2. * np.pi) # returns 200,200
    in_attenuation_length = sim.attenuation_length(sim.source.pos, point)
    segment_centers = math2d.center(sim.detector.segments)
    out_attenuation_lengths = np.array([sim.attenuation_length(point, center) for center in segment_centers])

    prob = np.exp(-in_attenuation_length) * np.multiply(detector_solid_angle, np.exp(-out_attenuation_lengths))

    return prob


def main():
    sim = build_shielded_geometry(True)

    plt.figure()

    sim.rotate(20.)
    sim.draw()

    print sim.source.angle


    angles = np.linspace(-15., 15., 50) * np.pi / 180.
    r = 50.
    fission_probs = np.zeros((len(angles), len(sim.detector.segments)))

    for i, angle in enumerate(angles):
        print i
        end = sim.source.pos + np.array([r * np.cos(angle - sim.source.angle / 2.), r * np.sin(angle - sim.source.angle / 2.)])

        segments, cross_sections = sim.fission_segments(sim.source.pos, end)

        for segment in segments:
            plt.plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], color='black')

        fission_probs[i, :] = propagate_fission_ray(sim, sim.source.pos, end, n=5)

    print np.max(fission_probs)
    print np.unravel_index(fission_probs.argmax(), fission_probs.shape)

    plt.figure()
    plt.imshow(fission_probs.T, extent=[-15., 15., -15., 15.], interpolation='none')
    plt.colorbar()
    plt.xlabel('Neutron Angle')
    plt.ylabel('Detector Bin Angle')
    plt.title('Single Fission Detection Probability (Arb. Z Scale)')

    plt.show()

if __name__ == "__main__":
    sys.exit(int(main() or 0))