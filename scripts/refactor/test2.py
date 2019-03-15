"""
[x] Fix bug in siddon for flat lines
[x] get Sinograms working
[x] create one pixel response
[x] create response matrix
[ ] reconstruct transmission, fission mu
[ ] convert fission response

[ ] Calculate forward response data
"""


import numpy as np
from numba import jit


import assemblies
from utils import Data
import raytrace
import algorithms
from math import sqrt, acos, fabs, exp, floor
from raytrace_c import c_bilinear_interpolation, fission_probability


nu_u235_induced = \
    np.array([0.0237898, 0.1555525, 0.3216515, 0.3150433, 0.1444732, 0.0356013, 0.0034339, 0.0004546])

nu_dist = nu_u235_induced


def draw_algorithm(extent, image, draw_rays=True):

    plt.imshow(image.T, extent=extent, origin='lower')

    if draw_rays:
        # vertical lines
        for i in range(np.size(image, 0) + 1):
            x = extent[0] + (extent[1] - extent[0]) / np.size(image, 0) * i
            plt.plot([x, x], [extent[2], extent[3]], 'g')

        # horizontal lines
        for i in range(np.size(image, 1) + 1):
            y = extent[2] + (extent[3] - extent[2]) / np.size(image, 1) * i
            plt.plot([extent[0], extent[1]], [y, y], 'g')


def draw_rays(rays, draw_option='b-'):
    for ray in rays:
        plt.plot([ray[0], ray[2]], [ray[1], ray[3]], draw_option, lw=1)


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


def arc_detector_points(center_x, center_y, radius, arc_angle, n_points, midpoint=False):

    points = np.zeros((n_points, 2), dtype=np.double)
    arc_radian = arc_angle / 180. * np.pi

    if midpoint:
        radian_edges = np.linspace(arc_radian/2, -arc_radian/2, n_points+1)
        radians = (radian_edges[1:] + radian_edges[:-1]) / 2.
    else:
        radians = np.linspace(arc_radian/2, -arc_radian/2, n_points)

    points[:, 0] = center_x + np.cos(radians) * radius
    points[:, 1] = center_y + np.sin(radians) * radius

    return points


# def solid_angle_line(line, point):
#
#     a = (line[0] - point[0]) * (line[0] - point[0]) + (line[1] - point[1]) * (line[1] - point[1])
#     b = (line[2] - point[0]) * (line[2] - point[0]) + (line[3] - point[1]) * (line[3] - point[1])
#     c = (line[0] - line[2]) * (line[0] - line[2]) + (line[1] - line[3]) * (line[1] - line[3])
#
#     num = a + b - c
#     denom = 2 * sqrt(a) * sqrt(b)
#     angle = acos(fabs(num / denom))
#     if angle > np.pi / 2.:
#         angle = np.pi - angle
#
#     return angle


def parallel_rays(minx, maxx, miny, maxy, n_rays):
    rays = np.zeros((n_rays, 4), dtype=np.double)
    rays[:, 0] = minx
    rays[:, 2] = maxx
    rays[:, 1] = np.linspace(miny, maxy, n_rays)
    rays[:, 3] = rays[:, 1]

    return rays


def fan_rays(radius, arc_angle, n_rays, angles=None, midpoint=False):
    if angles is None:
        rays = np.zeros((n_rays, 4), dtype=np.double)
        rays[:, (0, 1)] = np.array([-radius, 0])
        rays[:, (2, 3)] = arc_detector_points(-radius, 0, radius * 2, arc_angle, n_rays, midpoint)

        return rays

    rays = np.zeros((angles.shape[0], n_rays, 4), dtype=np.double)
    rays[:, :, (0, 1)] = np.array([-radius, 0])
    rays[:, :, (2, 3)] = arc_detector_points(-radius, 0, radius * 2, arc_angle, n_rays, midpoint)
    for i in range(angles.shape[0]):
        rays[i, :, :2] = rotate_points(rays[i, :, :2], angles[i])
        rays[i, :, 2:] = rotate_points(rays[i, :, 2:], angles[i])

    return rays


def transmission_project(rays, image, extent, step_size=1e-3):
    projection = raytrace.raytrace_bulk_bilinear(rays.reshape(-1, rays.shape[-1]), image, extent, step_size=step_size)
    return projection.reshape(rays.shape[0], rays.shape[1])


def transmission_backproject(rays, sinogram, image_shape, extent, step_size=1e-3):
    return raytrace.raytrace_backproject_bulk(rays.reshape(-1, rays.shape[-1]), sinogram.flatten(), image_shape, extent,
                                              step_size=step_size)


# Fission Algorithms
def fission_project(rays, k, mu_image, mu_f_image, p_image, extent, step_size=1e-3):
    pass


def fission_backproject(rays, k, mu_image, mu_f_image, p_image, extent, step_size=1e-3):
    pass


# def fission_probability(ray, k, mu_image, mu_f_image, p_image, extent, detector_points, step_size=1e-3):
#     # assumes rays originate outside of image boundaries defined by extent
#     ex1, ex2, ey1, ey2 = extent
#
#     line = raytrace.line_box_overlap_line(ray, extent)
#     line_distance = sqrt((line[2] - line[0]) ** 2 + (line[3] - line[1]) ** 2)
#     if line_distance == 0:
#         return
#
#     fission_prob_integral = 0
#     absorbance_in = 0.
#     n_steps = max(int(floor(line_distance / step_size)), 2)
#     step = line_distance / n_steps
#     pos = np.array([line[0], line[1]], dtype=np.double)
#
#     mu = c_bilinear_interpolation(pos[0], pos[1], mu_f_image, ex1, ex2, ey1, ey2)
#     mu_f = c_bilinear_interpolation(pos[0], pos[1], mu_f_image, ex1, ex2, ey1, ey2)
#     p = c_bilinear_interpolation(pos[0], pos[1], p_image, ex1, ex2, ey1, ey2)
#     if mu <= 0 or mu_f <= 0 or p <= 0:
#         fission_prob_prev = 0
#     else:
#         absorbance_in += 0.
#
#         enter_prob = exp(-absorbance_in)
#         detector_prob = detect_probability(pos, mu_image, extent, detector_points, step_size)
#         exit_prob = exit_probability(p, k, nu_dist, detector_prob)
#
#         mu_prev = mu
#         fission_prob_prev = enter_prob * mu_f * exit_prob
#
#     for i in range(n_steps - 1):
#         pos[0] = line[0] + (i+1) * (line[2] - line[0]) / n_steps
#         pos[1] = line[1] + (i+1) * (line[3] - line[1]) / n_steps
#
#         mu = c_bilinear_interpolation(pos[0], pos[1], mu_image, ex1, ex2, ey1, ey2)
#         if mu <= 0:
#             mu_prev = 0
#             continue
#         mu_f = c_bilinear_interpolation(pos[0], pos[1], mu_f_image, ex1, ex2, ey1, ey2)
#         if mu_f <= 0:
#             continue
#         p = c_bilinear_interpolation(pos[0], pos[1], p_image, ex1, ex2, ey1, ey2)
#         if p <= 0:
#             continue
#
#         absorbance_in += (mu_prev + mu) * (line_distance / n_steps / 2)
#
#         enter_prob = exp(-absorbance_in)
#         detector_prob = detect_probability(pos, mu_image, *extent, detector_points, step_size)
#         exit_prob = exit_probability(p, k, nu_dist, detector_prob)
#
#         fission_prob = enter_prob * mu_f * exit_prob
#
#         fission_prob_integral += (fission_prob + fission_prob_prev) * (line_distance / n_steps / 2)
#
#         mu_prev = mu
#         fission_prob_prev = fission_prob
#
#     return fission_prob_integral


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # create and save transmission pixel response
    """
    mu_im, mu_f_im, p_im = assemblies.shielded_true_images()

    # rays = parallel_rays(-15, 15, -12, 12, 100)
    rays = fan_rays(80, 40, 100*10)

    plt.figure()
    plt.imshow(mu_im.data, extent=mu_im.extent)

    response = image_response_sinogram(rays, mu_im, np.linspace(0., 180., 100), rays_downsample=10)

    np.save('response.npy', response)
    """
    # create object sinogram and display pixel responses
    """
    mu_im, mu_f_im, p_im = assemblies.shielded_true_images()
    plt.figure()
    plt.imshow(mu_im.data, extent=mu_im.extent)

    rays = fan_rays(80, 40, 100)
    mu_sinogram = rotation_sinogram(rays, mu_im, np.linspace(0., 180., 200))

    plt.figure()
    plt.imshow(mu_sinogram)

    response = np.load('response.npy')

    plt.figure()
    plt.imshow(response[100])

    plt.figure()
    plt.imshow(response[250])

    plt.figure()
    plt.imshow(response[411])

    plt.show()
    """
    # transmission - reconstruct object from object sinogram and response matrix
    """
    response = np.load('response.npy')

    mu_im, mu_f_im, p_im = assemblies.shielded_true_images()
    plt.figure()
    plt.imshow(mu_im.data, extent=mu_im.extent)

    rays = fan_rays(80, 40, 100)
    mu_sinogram = rotation_sinogram(rays, mu_im, np.linspace(0., 180., 100))

    plt.figure()
    plt.imshow(mu_sinogram)

    d = mu_sinogram.reshape((-1))
    G = response.reshape((response.shape[0], -1)).T

    m = algorithms.solve_tikhonov(d, G, 2.)
    print(m.shape)
    m = m.reshape((mu_im.data.shape[1], mu_im.data.shape[0]))

    plt.imshow(m.T, extent=mu_im.extent)

    plt.show()
    """
    # test line pixel index display is working
    """
    mu_im, mu_f_im, p_im = assemblies.shielded_true_images()
    mu_im.data[:, :] = 1
    plt.figure()
    plt.imshow(mu_im.data, extent=mu_im.extent)

    source_rays = fan_rays(80, 40, 25, midpoint=True)
    # line = np.array([-40., 0., 11.85, 7.26666667], dtype=np.double)
    line = np.array([4.0000000e+01, -4.8985872e-15, 6.7500000e+00, -3.8000000e+00], dtype=np.double)

    ans = pyraytrace(line, mu_im.extent, mu_im.data, debug=True)
    print(ans)
    # line = source_rays[12]
    pixels, distances = raytrace.raytrace_fast_store(line, mu_im.extent, mu_im.data)
    ans = raytrace.raytrace_fast(line, mu_im.extent, mu_im.data)
    print(ans)

    plt.plot([line[0], line[2]], [line[1], line[3]])

    for i in range(pixels.shape[0]):
        pixel_i = pixels[i, 0]
        pixel_j = pixels[i, 1]
        mu_image = mu_im.data
        extent = mu_im.extent
        pixel_x = pixel_i / (mu_image.shape[0]) * (extent[1] - extent[0]) + extent[0] + 0.5 * (extent[1] - extent[0]) / \
                  mu_image.shape[0]
        pixel_y = pixel_j / (mu_image.shape[1]) * (extent[3] - extent[2]) + extent[2] + 0.5 * (extent[3] - extent[2]) / \
                  mu_image.shape[1]
        # plt.scatter([pixel_x], [pixel_y])

    print(distances.sum())
    plt.show()
    # """
    # create fission sinogram
    """
    mu_im, mu_f_im, p_im = assemblies.shielded_true_images()

    plt.figure()
    plt.imshow(mu_im.data, extent=mu_im.extent)

    detector_points = arc_detector_points(-40, 0, 80, 40, 11)
    plt.scatter(detector_points[:, 0], detector_points[:, 1])
    plt.plot(detector_points[:, 0], detector_points[:, 1])

    source_rays = fan_rays(80, 40, 25, midpoint=True)
    draw_rays(source_rays)

    angles = np.linspace(0., 180., 25)

    mu_f_sinogram = fission_rotation_sinogram(source_rays, detector_points, angles, mu_im.extent,
                                              mu_im.data, mu_f_im.data, p_im.data, k=1)

    plt.figure()
    plt.imshow(mu_f_sinogram)

    plt.show()
    """
    # test bilinear
    """
    import cProfile

    mu_im, mu_f_im, p_im = assemblies.shielded_true_images()

    plt.figure()
    plt.imshow(mu_im.data, extent=mu_im.extent, origin='lower')

    detector_points = arc_detector_points(-40, 0, 80, 40, 11)
    plt.scatter(detector_points[:, 0], detector_points[:, 1])
    plt.plot(detector_points[:, 0], detector_points[:, 1])

    source_rays = fan_rays(80, 40, 200, midpoint=True)
    # draw_rays(source_rays)
    angles = np.linspace(0, 360, 200)

    pr = cProfile.Profile()
    pr.enable()
    sino = rotation_sinogram(source_rays, mu_im, angles, step_size=0.05)
    pr.disable()
    pr.print_stats(sort='time')

    plt.figure()
    # plt.plot(sino)
    plt.imshow(sino, extent=[angles[0], angles[-1], -20, 20], aspect='auto', interpolation='nearest')

    plt.show()
    """
    # test backproject
    """
    radius = 40
    detector_arc_angle = 40
    n_rays = 200
    n_angles = 200
    step_size = 0.05

    angles = np.linspace(0, 360, n_angles)
    rays = fan_rays(radius, detector_arc_angle, n_rays, angles, midpoint=True)

    mu_im, mu_f_im, p_im = assemblies.shielded_true_images()

    sino = transmission_project(rays, mu_im.data, mu_im.extent, step_size=step_size)
    backprojection = transmission_backproject(rays, sino, mu_im.data.shape, mu_im.extent, step_size=step_size)

    plt.figure()
    plt.imshow(mu_im.data, extent=mu_im.extent, origin='lower')
    draw_rays(rays[0])

    plt.figure()
    plt.imshow(sino, extent=[-detector_arc_angle/2, detector_arc_angle/2, angles[0], angles[-1]], aspect='auto', interpolation='nearest')

    plt.figure()
    plt.imshow(backprojection, extent=mu_im.extent, origin='lower')

    plt.show()
    """
    # Test Kaczmarz's Algorithm
    """
    radius = 40
    detector_arc_angle = 40
    n_rays = 200
    n_angles = 200
    step_size = 0.05

    angles = np.linspace(0, 360, n_angles)
    rays = fan_rays(radius, detector_arc_angle, n_rays, angles, midpoint=True)

    mu_im, mu_f_im, p_im = assemblies.shielded_true_images()

    sino = transmission_project(rays, mu_im.data, mu_im.extent, step_size=step_size)
    backprojection = transmission_backproject(rays, sino, mu_im.data.shape, mu_im.extent, step_size=step_size)

    def kaczmarz(rays, extent, sino, m0, step_size):
        for i in range(m0.shape[0]):
            m[i+1] = m[i] - transmission_project(rays[i+1], m[i], extent, step_size=step_size) - sino[i+1]
    """
    # Test Fission Sinogram
    radius = 40
    detector_arc_angle = 40
    n_rays = 200
    n_angles = 200
    step_size = 0.1

    angles = np.linspace(0, 360, n_angles)
    rays = fan_rays(radius, detector_arc_angle, n_rays, angles, midpoint=True)

    detector_points = arc_detector_points(-radius, 0, radius * 2, detector_arc_angle, n_rays, midpoint=False)

    mu_im, mu_f_im, p_im = assemblies.shielded_true_images()

    plt.figure()
    plt.imshow(mu_im.data, extent=mu_f_im.extent, origin='lower')
    # draw_rays(rays[0])
    plt.figure()
    plt.imshow(mu_f_im.data, extent=mu_f_im.extent, origin='lower')
    plt.figure()
    plt.imshow(p_im.data, extent=p_im.extent, origin='lower')

    rays_ = rays.reshape(-1, rays.shape[-1])
    sinogram = np.zeros(rays_.shape[0], dtype=np.double)
    for i in range(rays_.shape[0]):
        if i % 100 == 0:
            print(i, ' / ', rays_.shape[0])
        sinogram[i] = fission_probability(rays_[i], 2, mu_im.data, mu_f_im.data, p_im.data, mu_im.extent,
                                          detector_points, nu_dist, step_size=0.05)

    sinogram = sinogram.reshape(rays.shape[0], rays.shape[1])

    plt.figure()
    plt.imshow(sinogram, extent=[-detector_arc_angle / 2, detector_arc_angle / 2, angles[0], angles[-1]], aspect='auto',
               interpolation='nearest')

    backprojection = transmission_backproject(rays, sinogram, mu_im.data.shape, mu_im.extent, step_size=step_size)
    plt.figure()
    plt.imshow(backprojection, extent=mu_im.extent, origin='lower')

    plt.show()