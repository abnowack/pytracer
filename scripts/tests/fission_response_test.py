import sys
import matplotlib.pyplot as plt
import numpy as np
from scripts.assemblies import shielded_assembly
import pytracer.geometry as geo
import pytracer.transmission as transmission
import pytracer.algorithms as algorithms
import pytracer.fission as fission

if __name__ == "__main__":
    assembly_solids = shielded_assembly()
    assembly_flat = geo.flatten(assembly_solids)

    plt.figure()
    geo.draw(assembly_solids)

    radians = np.linspace(0, np.pi, 100)
    arc_radians = np.linspace(-np.pi / 8, np.pi / 8, 100)
    source, detector_points, extent = geo.fan_beam_paths(60, arc_radians, radians, extent=True)
    source = source[0, :, :]

    # grid = geo.Grid(width=25, height=15, num_x=50, num_y=30)
    grid = geo.Grid(width=25, height=15, num_x=25, num_y=15)
    grid.draw()

    cell_i = 109
    unit_m = geo.Material('black', 1, 1)
    vacuum = geo.Material('white', 0, 0)
    grid_points = grid.cell(cell_i)
    cell_geom = [geo.Solid(geo.convert_points_to_segments(grid_points, circular=True), unit_m, vacuum)]
    cell_flat = geo.flatten(cell_geom)
    plt.scatter(grid_points[:, 0], grid_points[:, 1], zorder=12)

    # display example orientation
    disp_angle = 53
    source_disp_angle = 27
    plt.scatter(source[disp_angle, 0], source[disp_angle, 1])
    plt.plot(detector_points[:, disp_angle, 0], detector_points[:, disp_angle, 1])
    for idisp_angle in range(len(detector_points)):
        plt.plot([source[disp_angle, 0], detector_points[idisp_angle, disp_angle, 0]],
                 [source[disp_angle, 1], detector_points[idisp_angle, disp_angle, 1]], color='blue', ls='dotted')
    plt.title('Detector Angle {d:.2f}, Neutron Angle {n:.2f}'.format(d=np.rad2deg(radians[disp_angle]),
                                                                     n=np.rad2deg(arc_radians[source_disp_angle])))

    detector_segments = geo.convert_points_to_segments(detector_points[:, disp_angle, :])

    probs = np.zeros(np.size(detector_points, 0))
    grid_probs = np.zeros(np.size(detector_points, 0))

    start = source[disp_angle]
    for i in range(len(probs)):
        end = detector_points[i, disp_angle]
        probs[i] = fission.probability_path_neutron(start, end, assembly_flat, detector_segments, 1, 0.2)
        grid_probs[i] = fission.grid_cell_response(start, end, detector_segments, cell_flat, assembly_flat, 1, 0.2)
        # segments, values = fission.find_fission_segments(start, end, cell_flat)
        # if len(segments) == 0:
        #     continue
        # segment = segments[0]
        # print(segment)
        #
        # fission_positions = fission.break_segment_into_points(segment, num_points=5)
        # plt.plot(segment[:, 0], segment[:, 1], zorder=15, color='red')
        # plt.scatter(fission_positions[:, 0], fission_positions[:, 1], zorder=16, color='red')
        #
        # length = np.sqrt((segment[1, 0] - segment[0, 0]) ** 2 + (segment[1, 1] - segment[0, 1]) ** 2)
        #
        # for j, position in enumerate(fission_positions):
        #     absorbance = transmission.find_absorbance_at_point(position, assembly_flat)
        #     probs[i] += fission.probability_per_ds_neutron_single(start, position, assembly_flat, absorbance,
        #                                                           detector_segments,
        #                                                           0.2)
        #     absorbance_in[i] = transmission.absorbance(start, position, assembly_flat.segments, assembly_flat.absorbance)
        #     prob_in[i] += fission.probability_in(start, position, assembly_flat)
        #     prob_out[i] += fission.probability_out_single(position, assembly_flat, detector_segments, 0.2)
        #
        # probs[i] /= length

    plt.figure()
    plt.plot(probs)
    plt.title('Probability')

    plt.figure()
    plt.plot(grid_probs)
    plt.title('Grid Probability')

    # plt.figure()
    # plt.plot(absorbance_in)
    # plt.title('Absorbance in')
    #
    # plt.figure()
    # plt.plot(prob_in)
    # plt.title('Prob in')
    #
    # plt.figure()
    # plt.plot(prob_out)
    # plt.title('Prob out')

    print(probs)

    plt.show()
