"""
Display fission segments
"""

import matplotlib.pyplot as plt
import numpy as np
from scripts.assemblies import shielded_assembly
import pytracer.geometry as geo
import pytracer.fission as fission

if __name__ == '__main__':
    assembly_solids = shielded_assembly()
    assembly_flat = geo.flatten(assembly_solids)

    print(assembly_flat.segments)

    plt.figure()
    geo.draw(assembly_solids, True)

    radians = np.linspace(0, np.pi, 10)
    arc_radians = np.linspace(-np.pi / 8, np.pi / 8, 10) / 4
    start, end, extent = geo.fan_beam_paths(60, arc_radians, radians, extent=True)
    for (s, e) in zip(start[:, 0], end[:, 0]):
        plt.plot([s[0], e[0]], [s[1], e[1]], color='blue', ls='dotted')

    for i, (s, e) in enumerate(zip(start[:, 0], end[:, 0])):
        segments, values = fission.find_fission_segments(s, e, assembly_flat)
        for (segment, value) in zip(segments, values):
            plt.plot(segment[:, 0], segment[:, 1], color='black', zorder=15)

    plt.show()
