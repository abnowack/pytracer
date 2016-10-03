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

    point = np.array([6., 2.])
    print(transmission.absorbance_at_point(point, assembly_flat))

    point = np.array([6., -2.])
    print(transmission.absorbance_at_point(point, assembly_flat))

    xs = np.linspace(-11, 11, 200)
    ys = np.linspace(-6, 6, 200)

    image, extent = transmission.absorbance_image(xs, ys, assembly_flat)

    plt.figure()
    plt.imshow(image.T, interpolation='none', extent=extent)
    plt.colorbar()
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')
    plt.show()