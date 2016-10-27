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
    geo.draw(assembly_solids, show_normals=True, fill=False)
    plt.plot([start[0], end[0]], [start[1], end[1]])

    assembly_flat = geo.flatten(assembly_solids)
    intercepts, indexes = transmission.intersections(start, end, assembly_flat.segments)

    for index in indexes:
        segment = assembly_flat.segments[index]
        plt.plot(segment[:, 0], segment[:, 1], color='black', lw=3)

    for intercept in intercepts:
        plt.scatter(intercept[0], intercept[1], s=100, facecolors='none', edgecolors='black', linewidths=2)

    plt.tight_layout()
    plt.show()
