import sys

import matplotlib.pyplot as plt
import numpy as np

from PyTracer.reconstruction import inverse_radon
from scripts.geometries import build_shielded_geometry


def main():
    sim = build_shielded_geometry()

    plt.figure()
    sim.draw(True)

    # plt.figure()
    # trace = sim.radon_transform(([11.01]))
    # plt.plot(trace[:, 0])
    # plt.show()

    n_angles = 200
    angles = np.linspace(0., 180., n_angles + 1)[:-1]

    radon = sim.radon_transform(angles, )

    plt.figure()
    plt.imshow(radon, cmap=plt.cm.Greys_r, interpolation='none', aspect='auto')
    plt.xlabel('Angle')
    plt.ylabel('Radon Projection')
    plt.colorbar()

    plt.figure()
    recon_image = inverse_radon(radon, angles)
    extent = [-sim.detector.width / 2., sim.detector.width / 2.]
    plt.imshow(recon_image.T[:, ::-1], interpolation='none', extent=extent *
    2)
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')
    plt.colorbar()

    plt.show()

if __name__ == "__main__":
    sys.exit(int(main() or 0))