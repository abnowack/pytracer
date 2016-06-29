import sys
import matplotlib.pyplot as plt
import numpy as np
from .assemblies import build_shielded_geometry
from pytracer.fission import propagate_fission_ray


def main():
    sim = build_shielded_geometry(True)

    # sim.rotate(20.)
    sim.draw()

    rays = sim.source.emit(-15, 15, 50)

    for ray in rays:
        plt.plot([sim.source.pos[0], sim.source.pos[0] + 50 * ray[0]],
                 [sim.source.pos[1], sim.source.pos[1] + 50 * ray[1]])

    fission_probs = np.zeros((np.size(rays, 0), sim.detector.nbins))

    for i, ray in enumerate(rays):
        end = sim.source.pos + 50 * ray

        segments, cross_sections = sim.geometry.fission_segments(sim.source.pos, end)

        for segment in segments:
            plt.plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], color='black')

        fission_probs[i, :] = propagate_fission_ray(sim, sim.source.pos, end, n=5)

    plt.figure()
    plt.imshow(fission_probs.T, extent=[-15., 15., -15., 15.], interpolation='none')
    plt.colorbar()
    plt.xlabel('Neutron Angle')
    plt.ylabel('Detector Bin Angle')
    plt.title('Single Fission Detection Probability (Arb. Z Scale)')

    plt.show()

if __name__ == "__main__":
    sys.exit(int(main() or 0))