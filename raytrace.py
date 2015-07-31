from mesh import create_hollow, translate_rotate_mesh, create_rectangle, create_circle, angle_matrix
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

def build_shielded_geometry():
    air = Material(0.1, color='white')
    u235_metal = Material(1.0, color='green')
    poly = Material(1.0, color='red')
    steel = Material(1.0, color='orange')

    box = create_hollow(create_rectangle(20., 10.), create_rectangle(18., 8.))

    hollow_circle = create_hollow(create_circle(3.9), create_circle(2.9))
    translate_rotate_mesh(hollow_circle, [-9 + 3.9 + 0.1, 0.])

    small_box_1 = create_rectangle(2., 2.)
    translate_rotate_mesh(small_box_1, [6., 2.])

    small_box_2 = create_rectangle(2., 2.)
    translate_rotate_mesh(small_box_2, [6., -2.])

    #sim = Simulation(air, 50., 45., 'arc')
    sim = Simulation(air, diameter=50., detector='arc')
    sim.detector.width = 30.
    sim.geometry.solids.append(Solid(box, steel, air))
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
    translate_rotate_mesh(box, rotate = angle_matrix(45.))

    sim = Simulation(air, diameter=50.)
    sim.detector.width = 30.
    sim.geometry.solids.append(Solid(ring, steel, air))
    sim.geometry.flatten()

    return sim

def main():
    sim = build_shielded_geometry()
    sim.detector.set_bins(100)

    plt.figure()
    sim.draw(True)

    #n_angles = 100
    #angles = np.linspace(0., 180., n_angles + 1)[:-1]

    #radon = sim.radon_transform(angles, nbins=200)

    #plt.figure()
    #plt.imshow(radon, cmap=plt.cm.Greys_r, interpolation='none',
    #aspect='auto')
    #plt.xlabel('Angle')
    #plt.ylabel('Radon Projection')
    #plt.colorbar()

    #plt.figure()
    #recon_image = inverse_radon(radon, angles)
    #extent = [-sim.detector.width / 2., sim.detector.width / 2.]
    #plt.imshow(recon_image.T[:, ::-1], interpolation='none', extent=extent * 2)
    #plt.xlabel('X (cm)')
    #plt.ylabel('Y (cm)')
    #plt.colorbar()

    plt.show()

if __name__ == "__main__":
    sys.exit(int(main() or 0))