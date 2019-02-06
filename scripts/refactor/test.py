"""
[ ] Fix line issue
[ ] get Sinograms working
[ ] create one pixel response
[ ] create response matrix
[ ] reconstruct transmission, fission mu
[ ] convert fission response
"""


import numpy as np

import assemblies
import utils
import raytrace


def parallel_rays(minx, maxx, miny, maxy, n_rays):
    rays = np.zeros((n_rays, 4), dtype=np.double)
    rays[:, 0] = minx
    rays[:, 2] = maxx
    rays[:, 1] = np.linspace(miny, maxy, n_rays)
    rays[:, 3] = rays[:, 1]

    return rays


def fan_rays(radius, arc_angle, n_rays):
    rays = np.zeros((n_rays, 4), dtype=np.double)
    rays[:, (0, 1)] = np.array([-radius / 2, 0])

    arc_radians = arc_angle / 180. * np.pi
    radians = np.linspace(arc_radians/2, -arc_radians/2, n_rays)
    rays[:, 2] = -radius / 2 + np.cos(radians) * radius
    rays[:, 3] = np.sin(radians) * radius

    return rays


def draw_rays(rays):
    for ray in rays:
        plt.plot([ray[0], ray[2]], [ray[1], ray[3]])


def sinogram(rays, image):
    return raytrace.raytrace_bulk_fast(rays, image.extent, image.data)


def rotate_points(points, angle, pivot=None, convert_to_radian=True):
    rotated_points = np.copy(points)
    if convert_to_radian:
        angle = angle / 180. * np.pi
    rotation_matrix = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])

    if pivot is not None:
        rotated_points -= pivot

    rotated_points = rotated_points @ rotation_matrix

    if pivot is not None:
        rotated_points += pivot

    return rotated_points


def rotate_rays(rays, angle, pivot=None):
    rotated_rays = np.copy(rays)
    rotated_rays[:, :2] = rotate_points(rotated_rays[:, :2], angle, pivot)
    rotated_rays[:, 2:] = rotate_points(rotated_rays[:, 2:], angle, pivot)
    return rotated_rays


def rotation_sinogram(rays, image, angles):

    result = np.zeros((angles.shape[0], rays.shape[0]), dtype=np.double)

    for i, angle in enumerate(angles):
        rotated_rays = rotate_rays(rays, angle)
        result[i] = sinogram(rotated_rays, image)

    return result


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    mu_im, mu_f_im, p_im = assemblies.shielded_true_images()

    rays = parallel_rays(-15, 15, -12, 12, 100)
    # rays = fan_rays(80, 40, 100)

    # measurement = rotation_sinogram(rays, mu_im, np.linspace(0., 180., 100))
    # plt.imshow(measurement)

    plt.imshow(mu_im.data, extent=mu_im.extent)
    draw_rays(rays)

    plt.show()
