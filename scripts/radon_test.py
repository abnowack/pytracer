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
    start, end = geo.radon_scan_paths(25, 100, 30, radians)
    for (s, e) in zip(start[:, 0], end[:, 0]):
        print(s, e)
        plt.plot([s[0], e[0]], [s[1], e[1]], color='blue')

    radon = transmission.scan(assembly_flat, start, end)

    plt.figure()
    plt.imshow(radon)

    # radon_scan_example(sim, 200)

    # response = build_transmission_response(sim, 100)
    # plt.figure()
    # plt.imshow(response[:, :, 0], interpolation='none')
    #
    # measurement, angles = radon(sim, 100)
    #
    # plt.figure()
    # plt.imshow(measurement, interpolation='none')
    #
    # for alpha in [500, 1000, 1500, 2000, 2500]:
    #     recon = recon_tikhonov(measurement, response, alpha=alpha)
    #     recon = recon.reshape(sim.grid.ny, sim.grid.nx)
    #     plt.figure()
    #     plt.imshow(recon, interpolation='none')
    #     plt.title('alpha = ' + str(alpha))

    plt.show()
