# -*- coding: utf-8 -*-
"""
Created on Fri Jun 05 11:50:01 2015

@author: Aaron

TODO: Implement Geometry Checking
    - Test if any lixels overlap
    - Cannot have hole in a hole, or solid in a solid
TODO: Use Bounded Volume Heirarchy to Reduce Lixel Search
TODO: Account for air attenuation by including outer material
      Currently will break if need to account for two materials in contact
"""
from mesh import create_hollow, translate_rotate_mesh, create_rectangle, create_circle, angle_matrix
from material import Material
from solid import Solid
from simulation import Simulation

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate
import sys

def inverse_radon(radon, thetas):
    pad_value = int(2 ** (np.ceil(np.log(2 * np.size(radon, 0)) / np.log(2))))

    f = np.fft.fftfreq(pad_value)
    ramp_filter = 2. * np.abs(f)

    reconstruction_image = np.zeros((np.size(radon, 0), np.size(radon, 0)))

    for i, theta in enumerate(thetas):
        filtered = np.real(np.fft.ifft(np.fft.fft(radon[:, i], n=pad_value) * ramp_filter))[:np.size(radon, 0)]
        back_projection = rotate(np.tile(filtered, (np.size(radon, 0), 1)), theta, reshape=False, mode='nearest')
        reconstruction_image += back_projection

    return reconstruction_image

def main():
    air = Material(0.0, 'white')
    u235_metal = Material(1.0, 'green')
    poly = Material(1.0, 'red')
    steel = Material(1.0, 'orange')

    box = create_hollow(create_rectangle(20., 10.), create_rectangle(18., 8.))

    hollow_circle = create_hollow(create_circle(3.9), create_circle(2.9))
    translate_rotate_mesh(hollow_circle, [-9 + 3.9 + 0.1, 0.])

    small_box_1 = create_rectangle(2., 2.)
    translate_rotate_mesh(small_box_1, [6., 2.])

    small_box_2 = create_rectangle(2., 2.)
    translate_rotate_mesh(small_box_2, [6., -2.])

    #sim = Simulation(air, 50., 45., 'arc')
    sim = Simulation(air)
    sim.detector.width = 40.
    sim.geometry.solids.append(Solid(box, steel, air))
    sim.geometry.solids.append(Solid(hollow_circle, poly, air))
    sim.geometry.solids.append(Solid(small_box_1, u235_metal, air))
    sim.geometry.solids.append(Solid(small_box_2, u235_metal, air))
    sim.geometry.flatten()

    plt.figure()
    sim.draw()

    n_angles = 100
    angles = np.linspace(0. ,180., n_angles + 1)[:-1]

    radon = sim.radon_transform(angles, nbins=200)

    plt.figure()
    plt.imshow(radon, cmap=plt.cm.Greys_r, interpolation='none', aspect='auto')

    plt.figure()
    recon_image = inverse_radon(radon, angles)
    plt.imshow(recon_image.T, cmap=plt.cm.Greys_r, interpolation='none')

    plt.show()

if __name__ == "__main__":
    sys.exit(int(main() or 0))