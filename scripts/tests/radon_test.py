import sys
import matplotlib.pyplot as plt
import numpy as np
from scripts.assemblies import shielded_assembly
import pytracer.geometry as geo
import pytracer.transmission as transmission
import cProfile, pstats

# 1    1.354    1.354    1.354    1.354 {built-in method pytracer.transmission_c.attenuations}
# after using distance function
# 1    1.030    1.030    1.030    1.030 {built-in method pytracer.transmission_c.attenuations}

if __name__ == "__main__":
    assembly_solids = shielded_assembly()
    assembly_flat = geo.flatten(assembly_solids)

    plt.figure()
    geo.draw(assembly_solids)

    radians = np.linspace(0, np.pi, 1000)
    start, end = geo.parallel_beam_paths(25, 1000, 30, radians)
    for (s, e) in zip(start[:, 0], end[:, 0]):
        plt.plot([s[0], e[0]], [s[1], e[1]], color='blue')

    pr = cProfile.Profile()
    pr.enable()
    radon = transmission.scan(assembly_flat, start, end)
    pr.disable()

    plt.figure()
    plt.imshow(radon)

    pr.print_stats()
    plt.show()
