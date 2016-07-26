import matplotlib.pyplot as plt
import numpy as np
from scripts.assemblies import shielded_assembly
import pytracer.geometry as geo
import pytracer.transmission as transmission

if __name__ == "__main__":
    assembly_solids = shielded_assembly()
    start = np.array([-20., -10.])
    end = np.array([20., 10.])

    plt.figure()
    geo.draw(assembly_solids, show_normals=True)
    plt.plot([start[0], end[0]], [start[1], end[1]])

    assembly_flat = geo.flatten(assembly_solids)
    intercepts, indexes = transmission.intersections(start, end, assembly_flat.segments)

    for intercept in intercepts:
        plt.plot(intercept[0], intercept[1], 'o', color='green')

    plt.tight_layout()
    plt.show()
