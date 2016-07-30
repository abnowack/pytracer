import sys
import matplotlib.pyplot as plt
import numpy as np
from scripts.assemblies import shielded_assembly
import pytracer.geometry as geo
import pytracer.fission as fission

if __name__ == '__main__':
    assembly_solids = shielded_assembly()
    assembly_flat = geo.flatten(assembly_solids)

    plt.figure()
    geo.draw(assembly_solids)

    # must create neutron beam paths and detector bins
    radians = np.linspace(0, np.pi, 100)
    arc_radians = np.linspace(-np.pi / 8, np.pi / 8, 100)
    source, detector_points, extent = geo.fan_beam_paths(60, arc_radians, radians, extent=True)
    source = source[0, :, :]

    # display example orientation
    disp_angle = 3
    source_disp_angle = 27
    plt.scatter(source[disp_angle, 0], source[disp_angle, 1])
    plt.plot(detector_points[:, disp_angle, 0], detector_points[:, disp_angle, 1])
    plt.plot([source[disp_angle, 0], detector_points[source_disp_angle, disp_angle, 0]],
             [source[disp_angle, 1], detector_points[source_disp_angle, disp_angle, 1]])
    plt.title('Detector Angle {d:.2f}, Neutron Angle {n:.2f}'.format(d=np.rad2deg(radians[disp_angle]),
                                                                     n=np.rad2deg(arc_radians[source_disp_angle])))

    # calculate singles and doubles measurement scans
    single_probs = fission.scan_single(source, detector_points, detector_points, assembly_flat, 0.2)
    double_probs = fission.scan_double(source, detector_points, detector_points, assembly_flat, 0.2)

    plt.figure()
    plt.imshow(single_probs.T, interpolation='none', extent=extent)
    plt.colorbar()
    plt.title('Single Neutron Probability')
    plt.xlabel('Detector Orientation')
    plt.ylabel('Relative Neutron Angle')
    plt.tight_layout()

    plt.figure()
    plt.imshow(double_probs.T, interpolation='none', extent=extent)
    plt.colorbar()
    plt.title('Double Neutron Probability')
    plt.xlabel('Detector Orientation')
    plt.ylabel('Relative Neutron Angle')
    plt.tight_layout()

    plt.show()
