from mesh import create_hollow, translate_rotate_mesh, create_rectangle, create_circle, angle_matrix
from material import Material
from solid import Solid
from simulation import Simulation

import numpy as np
import matplotlib.pyplot as plt
import sys

def plot_macro_fission(sim, start, end):
    start_points, end_points, macro_fissions = sim.fission_segments(start, end)
    print start_points
    print end_points
    for i in xrange(len(start_points)):
        start_point = start_points[i]
        end_point = end_points[i]
        macro_fission = macro_fissions[i]
        start_distance = np.sqrt((start_point[0] - start[0]) ** 2 + (start_point[1] - start[1]) ** 2)
        end_distance = np.sqrt((end_point[0] - start[0]) ** 2 + (end_point[1] - start[1]) ** 2)
        plt.plot([start_distance, end_distance], [macro_fission, macro_fission])

def build_shielded_geometry():
    air = Material(0.0, color='white')
    u235_metal = Material(0.2, 0.1, color='green')
    poly = Material(0.2, color='red')
    steel = Material(0.2, color='orange')

    box = create_hollow(create_rectangle(20., 10.), create_rectangle(18., 8.))

    hollow_circle = create_hollow(create_circle(3.9), create_circle(2.9))
    translate_rotate_mesh(hollow_circle, [-9 + 3.9 + 0.1, 0.])

    small_box_1 = create_rectangle(2., 2.)
    translate_rotate_mesh(small_box_1, [6., 2.])

    small_box_2 = create_rectangle(2., 2.)
    translate_rotate_mesh(small_box_2, [6., -2.])

    sim = Simulation(air, 50., 45., 'arc')
    #sim = Simulation(air, diameter=50.,)
    sim.detector.width = 30.
    sim.geometry.solids.append(Solid(box, steel, air))
    sim.geometry.solids.append(Solid(hollow_circle, u235_metal, air))
    sim.geometry.solids.append(Solid(small_box_1, poly, air))
    sim.geometry.solids.append(Solid(small_box_2, steel, air))
    sim.geometry.flatten()

    return sim

def ray_trace_test_geometry():
    air = Material(0.0, color='white')
    steel = Material(0.2, color='red')

    box = create_hollow(create_rectangle(12., 12.), create_rectangle(10., 10.))
    ring = create_hollow(create_circle(12.), create_circle(10.))
    translate_rotate_mesh(box, rotate = angle_matrix(45.))

    sim = Simulation(air, diameter=50.)
    sim.detector.width = 30.
    sim.geometry.solids.append(Solid(ring, steel, air))
    sim.geometry.flatten()

    return sim

def propogate_fission_ray(sim, start, end, n):
    segments, macro_fissions = sim.fission_segments(start, end)
    segment_probs = []
    for i in xrange(len(segments)):
        single_fission_prob = propogate_fissions_segment(sim, segments[i], macro_fissions[i], n)
        segment_probs.append(single_fission_prob)
    total_fission_prob = np.sum(segment_probs, axis=0)
    return total_fission_prob

def propogate_fissions_segment(sim, segment, macro_fission, n=5):
    point_0, point_1 = segment[0], segment[1]
    # generate points along fission segment
    # use trapezoid rule on uniform spacing
    # int [f(x = [a, b]) dx]  ~= (b - a) / (2 * N) [ f(a) + f(b) +  ] 
    points = [point_0 + (point_1 - point_0) * t for t in np.linspace(0.01, 0.99, n)] # TODO : error if t = 1
    values = np.zeros((len(points), len(sim.detector.bin_centers)))
    integral = np.zeros((len(sim.detector.bin_centers)))
    #values = np.zeros((len(sim.detector.bin_centers), len(points)))
    for i in xrange(len(points)):
        values[i, :] = propogate_fissions_point_detector(sim, points[i], macro_fission)
    integral[:] = np.linalg.norm(point_1 - point_0) / (n - 1) * (values[0, :] + 2. * np.sum(values[1:-1, :], axis=0) + values[-1, :])
    return integral 

def propogate_fissions_point_detector(sim, point, macro_fission):
    """
    Calculate probability of induced fission being detected over detector plane.

    nu = 1 for now, not using macro_fission
    """
    detector_solid_angle = sim.detector.solid_angles(point) / (2. * np.pi)
    in_attenuation_length = sim.attenuation_length(sim.source, point)
    out_attenuation_lengths = np.array([sim.attenuation_length(point, bin_c) for bin_c in sim.detector.bin_centers])

    prob = np.exp(-in_attenuation_length) * np.multiply(detector_solid_angle, np.exp(-out_attenuation_lengths))

    return prob

def main():
    sim = build_shielded_geometry()
    sim.detector.set_bins(100)

    plt.figure()
    sim.draw(False)

    angles = np.linspace(-15., 15., 20) * np.pi / 180.
    r = 50.
    start = sim.source
    fission_probs = np.zeros((len(angles), len(sim.detector.bin_centers)))

    for i, angle in enumerate(angles):
        print i
        end = start + np.array([r * np.cos(angle), r * np.sin(angle)])

        segments, cross_sections = sim.fission_segments(start, end)

        for segment in segments:
            plt.plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], color='black')

        fission_probs[i, :] = propogate_fission_ray(sim, start, end, n=5)

    print np.max(fission_probs)
    print np.unravel_index(fission_probs.argmax(), fission_probs.shape)

    plt.figure()
    plt.imshow(fission_probs.T)
    plt.colorbar()
    #for i in xrange(np.size(fission_probs, 0)):
    #    plt.plot(fission_probs[i, :])

    plt.show()

if __name__ == "__main__":
    main()
    #sys.exit(int(main() or 0))