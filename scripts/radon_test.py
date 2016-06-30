import sys
import matplotlib.pyplot as plt
import numpy as np
from scripts.assemblies import shielded_assembly
import pytracer.geometry as geo
import pytracer.transmission as transmission

if __name__ == "__main__":
    assembly_solids = shielded_assembly()
    assembly_flat = geo.flatten(assembly_solids)

    plt.figure()
    geo.draw(assembly_solids)

    radians = np.linspace(0, np.pi, 100)
    start, end = geo.parallel_beam_paths(25, 100, 30, radians)
    for (s, e) in zip(start[:, 0], end[:, 0]):
        print(s, e)
        plt.plot([s[0], e[0]], [s[1], e[1]], color='blue')

    radon = transmission.scan(assembly_flat, start, end)

    plt.figure()
    plt.imshow(radon)

    plt.show()
