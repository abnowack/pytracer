import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scripts.assemblies import shielded_assembly
import pytracer.geometry as geo
import pytracer.fission as fission
import cProfile, pstats

if __name__ == '__main__':
    assembly_solids = shielded_assembly()
    assembly_flat = geo.flatten(assembly_solids)

    plt.figure()
    # geo.draw(assembly_solids)

    # must create neutron beam paths and detector bins
    radians = np.linspace(0, np.pi, 100)
    arc_radians = np.linspace(-np.pi / 8, np.pi / 8, 100)
    source, detector_points, extent = geo.fan_beam_paths(60, arc_radians, radians, extent=True)
    source = source[0, :, :]

    # display example orientation
    disp_angle = 10
    source_disp_angle = 25
    plt.scatter(source[disp_angle, 0], source[disp_angle, 1], color='green', s=50)
    d, = plt.plot(detector_points[:, disp_angle, 0], detector_points[:, disp_angle, 1], lw=2.5)
    ax = plt.axes()
    l1 = ax.arrow(source[disp_angle, 0], source[disp_angle, 1],
                  detector_points[source_disp_angle, disp_angle, 0] - source[disp_angle, 0],
                  detector_points[source_disp_angle, disp_angle, 1] - source[disp_angle, 1], shape='full',
                  length_includes_head=True, width=0.1, color='green')
    # l1, = plt.plot([source[disp_angle, 0], detector_points[source_disp_angle, disp_angle, 0]],
    #          [source[disp_angle, 1], detector_points[source_disp_angle, disp_angle, 1]], 'g', lw=2.5)
    l2, = plt.plot([source[disp_angle, 0], detector_points[len(arc_radians) / 2, disp_angle, 0]],
                   [source[disp_angle, 1], detector_points[len(arc_radians) / 2, disp_angle, 1]], '--', color='gray',
                   lw=2.5, zorder=5)
    l3, = plt.plot([-source[0, 0] * 0.65, 0],
                   [source[0, 1], 0], '--', color='gray', lw=2.5)
    ax = plt.gca()

    x1, y1 = [[source[disp_angle, 0], detector_points[source_disp_angle, disp_angle, 0]],
              [source[disp_angle, 1], detector_points[source_disp_angle, disp_angle, 1]]]
    x2, y2 = l2.get_data()

    t1 = np.arctan2(y1[1] - y1[0], x1[1] - x1[0]) * 180 / np.pi
    t2 = np.arctan2(y2[1] - y2[0], x2[1] - x2[0]) * 180 / np.pi
    arc = patches.Arc(xy=(x1[0], y1[0]), width=50, height=50, theta1=t1, theta2=t2)
    arc.set_linewidth(2.5)
    ax.add_patch(arc)
    ax.text(-5, -2.5, r'$\phi$', fontsize=20)

    t1 = np.arctan2(y1[1] - 0, x1[1] - 0) * 180 / np.pi
    t2 = np.arctan2(y2[1] - 0, x2[1] - 0) * 180 / np.pi
    arc = patches.Arc(xy=(0, 0), width=40, height=40, theta1=t2, theta2=0)
    arc.set_linewidth(2.5)
    ax.add_patch(arc)
    ax.text(21, -4, r'$\theta$', fontsize=20)

    ax.text(-38, 11, "Neutron Generator (A)", fontsize=12, color='green')
    ax.text(-2, -22, "Neutron Detection (B)", fontsize=12, color='green')
    ax.text(13, 10, "Detector Array", fontsize=12, color='blue')

    plt.title(
        r'Detector Angle $\theta = $ {d:.2f} (rad), Neutron Angle $\phi = $ {n:.2f} (rad)'.format(d=radians[disp_angle],
                                                                                 n=arc_radians[source_disp_angle]))

    d.set_zorder(50)

    # calculate singles and doubles measurement scans

    # prorfile time
    # pr = cProfile.Profile()
    # pr.enable()
    # single_probs = fission.scan(source, detector_points, detector_points, assembly_flat, 1, 0.2)
    # double_probs = fission.scan(source, detector_points, detector_points, assembly_flat, 2, 0.2)
    # pr.disable()
    #
    # plt.figure()
    # plt.imshow(single_probs.T, interpolation='none', extent=extent)
    # # plt.colorbar()
    # plt.title('Single Neutron Probability')
    # plt.xlabel('Detector Orientation')
    # plt.ylabel('Relative Neutron Angle')
    # plt.tight_layout()
    #
    # plt.figure()
    # plt.imshow(double_probs.T, interpolation='none', extent=extent)
    # # plt.colorbar()
    # plt.title('Double Neutron Probability')
    # plt.xlabel('Detector Orientation')
    # plt.ylabel('Relative Neutron Angle')
    # plt.tight_layout()

    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    plt.tight_layout()
    plt.show()

    # pr.print_stats(sort='time')
