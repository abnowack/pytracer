import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scripts.assemblies import shielded_assembly
import pytracer.geometry as geo
import pytracer.transmission as transmission
import pytracer.algorithms as algorithms

if __name__ == "__main__":
    assembly_solids = shielded_assembly()
    assembly_flat = geo.flatten(assembly_solids)

    radians = np.linspace(0, np.pi, 100)
    arc_radians = np.linspace(-np.pi / 8, np.pi / 8, 100)
    source, detector_points, extent = geo.fan_beam_paths(60, arc_radians, radians, extent=True)

    measurement = transmission.scan(assembly_flat, source, detector_points)

    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    im = ax.imshow(np.exp(-measurement), interpolation='none', extent=extent, cmap='viridis')
    plt.title('Transmission Sinogram', size=20)
    plt.xlabel(r'Detector Angle $\theta$ (rad)', size=18)
    plt.ylabel(r'Neutron Angle $\phi$ (rad)', size=18)
    # cb = plt.colorbar(im, fraction=0.046, pad=0.04)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(r'$P_{transmission}$', size=18, labelpad=15)

    plt.tight_layout()

    plt.show()
