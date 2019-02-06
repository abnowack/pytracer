import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scripts.assemblies import shielded_true_image
from scripts.utils import display_estimate


def display_transmission_sinogram():
    import numpy as np
    from pytracer import geometry as geo



    radians = np.linspace(0, np.pi, 100)
    arc_radians = np.linspace(-np.pi / 8, np.pi / 8, 100)
    source, detector_points, extent = geo.fan_beam_paths(60, arc_radians, radians, extent=True)

    measurement = transmission.scan(assembly_flat, source, detector_points)

    plt.figure(figsize=(9, 6))
    ax = plt.gca()
    im = ax.imshow(np.exp(-measurement), interpolation='none', extent=extent, cmap='viridis')
    plt.title('Transmission Sinogram', size=20)
    plt.xlabel(r'Detector Angle $\theta$ (rad)', size=18)
    plt.ylabel(r'Neutron Angle $\phi$ (rad)', size=18)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(r'$P_{transmission}$', size=18, labelpad=15)

    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    true_image = shielded_true_image()

    display_estimate(true_image)
    plt.show()

    display_transmission_sinogram()