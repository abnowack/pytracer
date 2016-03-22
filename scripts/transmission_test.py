import sys

import matplotlib.pyplot as plt
import numpy as np

from PyTracer.transmission import radon, inverse_radon
from geometries import build_shielded_geometry


def main():
    sim = build_shielded_geometry()

    # plt.figure()
    sim.rotate(-45.)
    sim.draw()
    plt.tight_layout()

    # plt.figure()
    # trace = sim.radon_transform(([11.01]))
    # plt.plot(trace[:, 0])

    n_angles = 200
    angles = np.linspace(0., 180., n_angles + 1)[:-1]

    r = radon(sim, angles)

    plt.figure()
    plt.imshow(r, cmap=plt.cm.Greys_r, interpolation='none', aspect='auto')
    plt.xlabel('Angle')
    plt.ylabel('Radon Projection')
    plt.colorbar()

    plt.figure()
    recon_image = inverse_radon(r, angles)
    extent = [-sim.detector.width / 2., sim.detector.width / 2.]
    plt.imshow(recon_image.T[::-1, :], interpolation='none', extent=extent * 2)
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')
    plt.colorbar()

    plt.show()


if __name__ == "__main__":
    sys.exit(int(main() or 0))
