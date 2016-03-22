import sys
import matplotlib.pyplot as plt
import numpy as np

from geometries import build_shielded_geometry
from PyTracer.fission import propagate_fission_ray

def main():
    sim = build_shielded_geometry(True)

    sim.rotate(20.)
    sim.draw()

    angles = np.linspace(-15, 15, 50)
    angles *= np.pi / 180.
    for angle in angles:
        plt.plot([sim.source.pos[0], sim.source.pos[0] + 10. * np.cos(np.pi * sim.source.angle / 180. - angle)],
                 [sim.source.pos[1], sim.source.pos[1] + 10. * np.sin(np.pi * sim.source.angle / 180. - angle)])

    plt.plot([sim.source.pos[0], - sim.source.pos[0]], [sim.source.pos[1], - sim.source.pos[1]])

    # angles = np.linspace(-15., 15., 50)
    # angles *= np.pi / 180.
    #
    # r = 50.
    # fission_probs = np.zeros((len(angles), sim.detector.nbins))

    # for i, angle in enumerate(angles):
    #     print i
    #     end = sim.source.pos + np.array([r * np.cos(angle - sim.source.angle / 2.), r * np.sin(angle - sim.source.angle / 2.)])

    #     segments, cross_sections = sim.geometry.fission_segments(sim.source.pos, end)
    #
    #     for segment in segments:
    #         plt.plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], color='black')
    #
    #     fission_probs[i, :] = propagate_fission_ray(sim, sim.source.pos, end, n=5)
    #
    # print np.max(fission_probs)
    # print np.unravel_index(fission_probs.argmax(), fission_probs.shape)

    # plt.figure()
    # plt.imshow(fission_probs.T, extent=[-15., 15., -15., 15.], interpolation='none')
    # plt.colorbar()
    # plt.xlabel('Neutron Angle')
    # plt.ylabel('Detector Bin Angle')
    # plt.title('Single Fission Detection Probability (Arb. Z Scale)')

    plt.show()

if __name__ == "__main__":
    sys.exit(int(main() or 0))