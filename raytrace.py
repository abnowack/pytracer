from mesh import create_hollow, create_rectangle, create_circle
from material import Material
from solid import Solid
from simulation import Simulation

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate
import sys


# TODO : Normalization isn't correct
def inverse_radon(radon, thetas):
    """
    Reconstruct using Filtered Back Projection.
    
    Weighting assumes thetas are equally spaced
    radon size must be even
    """
    pad_value = int(2 ** (np.ceil(np.log(2 * np.size(radon, 0)) / np.log(2))))
    pre_pad = int((pad_value - len(radon[:, 0])) / 2)
    post_pad = pad_value - len(radon[:, 0]) - pre_pad

    f = np.fft.fftfreq(pad_value)
    ramp_filter = 2. * np.abs(f)

    reconstruction_image = np.zeros((np.size(radon, 0), np.size(radon, 0)))

    for i, theta in enumerate(thetas):
        filtered = np.real(np.fft.ifft(np.fft.fft(np.pad(radon[:, i], (pre_pad, post_pad), 'constant', constant_values=(0, 0))) * ramp_filter))[pre_pad:-post_pad]
        back_projection = rotate(np.tile(filtered, (np.size(radon, 0), 1)), theta, reshape=False, mode='constant')
        reconstruction_image += back_projection * 2 * np.pi / len(thetas)

    return reconstruction_image


def build_shielded_geometry(fission=False):
    air = Material(0.001, color='white')
    u235_metal = Material(0.1, 0.1, color='green')
    poly = Material(0.05, color='red')
    steel = Material(0.07, color='orange')

    box = create_hollow(create_rectangle(20., 10.), create_rectangle(18., 8.))

    hollow_circle = create_hollow(create_circle(3.9), create_circle(2.9))
    hollow_circle.translate([-9 + 3.9 + 0.1, 0.])

    small_box_1 = create_rectangle(2., 2.)
    small_box_1.translate([6., 2.])

    small_box_2 = create_rectangle(2., 2.)
    small_box_2.translate([6., -2.])

    # sim = Simulation(air, 50., 45., 'arc')
    sim = Simulation(air, 200, diameter=50., detector='plane', detector_width=30.)
    sim.detector.width = 30.
    sim.geometry.solids.append(Solid(box, steel, air))
    if fission:
        sim.geometry.solids.append((Solid(hollow_circle, u235_metal, air)))
    else:
        sim.geometry.solids.append(Solid(hollow_circle, steel, air))
    sim.geometry.solids.append(Solid(small_box_1, poly, air))
    sim.geometry.solids.append(Solid(small_box_2, steel, air))
    sim.geometry.flatten()

    return sim


def ray_trace_test_geometry():
    air = Material(0.0, color='white')
    steel = Material(1.0, color='red')

    box = create_hollow(create_rectangle(12., 12.), create_rectangle(10., 10.))
    ring = create_hollow(create_circle(12.), create_circle(10.))
    box.rotate(45.)

    sim = Simulation(air, 100, diameter=50.)
    sim.detector.width = 30.
    sim.geometry.solids.append(Solid(ring, steel, air))
    sim.geometry.flatten()

    return sim


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