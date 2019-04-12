"""
[ ] Speed up Fission Projection
[ ] Get reconstruction algorithms working for fission sinograms
[ ] Get CGLS working
"""


import numpy as np
from numba import jit


import assemblies
from utils import Data
import raytrace
import algorithms
from math import sqrt, acos, fabs, exp, floor
from raytrace_c import c_bilinear_interpolation, fission_probability, c_backproject_parallel, c_backproject_fan


nu_u235_induced = \
    np.array([0.0237898, 0.1555525, 0.3216515, 0.3150433, 0.1444732, 0.0356013, 0.0034339, 0.0004546])

nu_dist = nu_u235_induced


def draw_rays(rays, draw_option='b-'):
    for ray in rays:
        plt.plot([ray[0], ray[2]], [ray[1], ray[3]], draw_option, lw=1)


def rotate_points(points, radian, pivot=None):
    rotated_points = np.copy(points)
    rotation_matrix = np.array([[np.cos(radian), np.sin(radian)], [-np.sin(radian), np.cos(radian)]])

    if pivot is not None:
        rotated_points -= pivot

    rotated_points = rotated_points @ rotation_matrix

    if pivot is not None:
        rotated_points += pivot

    return rotated_points


def rotate_rays(rays, radian, pivot=None):
    rotated_rays = np.copy(rays)
    rotated_rays[:, :2] = rotate_points(rotated_rays[:, :2], radian, pivot)
    rotated_rays[:, 2:] = rotate_points(rotated_rays[:, 2:], radian, pivot)
    return rotated_rays


def arc_detector_points(center_x, center_y, radius, arc_radian, n_points, midpoint=False):

    points = np.zeros((n_points, 2), dtype=np.double)

    if midpoint:
        radian_edges = np.linspace(arc_radian/2, -arc_radian/2, n_points+1)
        radians = (radian_edges[1:] + radian_edges[:-1]) / 2.
    else:
        radians = np.linspace(arc_radian/2, -arc_radian/2, n_points)

    points[:, 0] = center_x + np.cos(radians) * radius
    points[:, 1] = center_y + np.sin(radians) * radius

    return points


def parallel_rays(ss, radians, length):
    rays = np.zeros((len(radians), len(ss), 4), dtype=np.double)
    for j, radian in enumerate(radians):
        for i, s in enumerate(ss):
            rays[j, i, 0] = s * np.cos(radian) + length / 2. * np.sin(radian)
            rays[j, i, 1] = s * np.sin(radian) - length / 2. * np.cos(radian)
            rays[j, i, 2] = s * np.cos(radian) - length / 2. * np.sin(radian)
            rays[j, i, 3] = s * np.sin(radian) + length / 2. * np.cos(radian)

    return rays


def fan_rays(radius, radian, n_rays, radians, midpoint=False):
    rays = np.zeros((len(radians), n_rays, 4), dtype=np.double)
    rays[:, :, (0, 1)] = np.array([-radius, 0])
    rays[:, :, (2, 3)] = arc_detector_points(-radius, 0, radius * 2, radian, n_rays, midpoint)

    for i in range(len(radians)):
        rays[i, :, :2] = rotate_points(rays[i, :, :2], radians[i])
        rays[i, :, 2:] = rotate_points(rays[i, :, 2:], radians[i])

    return rays


def generate_rays(ray_geom):
    if ray_geom['type'] == 'fan':
        rays = fan_rays(ray_geom['radius'], ray_geom['detector_arc_radian'], ray_geom['n_rays'], ray_geom['radians'],
                        ray_geom['midpoint'])
    elif ray_geom['type'] == 'parallel':
        rays = parallel_rays(ray_geom['ss'], ray_geom['radians'], ray_geom['length'])
    else:
        rays = None

    return rays


def transmission_project(ray_geom, image, extent, step_size=1e-3, rays=None):
    if rays is None:
        rays = generate_rays(ray_geom)

    if rays.ndim > 2:
        projection = raytrace.raytrace_bulk_bilinear(rays.reshape(-1, rays.shape[-1]), image, extent, step_size=step_size)
        return projection.reshape(rays.shape[0], rays.shape[1])
    else:
        projection = raytrace.raytrace_bulk_bilinear(rays, image, extent, step_size=step_size)
        return projection


def padded_radians(ray_geom):
    # create rays with extra rays on the ends for the zero padding in the interpolation step
    dradian = ray_geom['detector_arc_radian'] / ray_geom['n_rays']

    pad_n_rays = ray_geom['n_rays'] + 2
    pad_arc_radian = dradian * pad_n_rays

    if ray_geom['midpoint']:
        radian_edges = np.linspace(pad_arc_radian / 2, -pad_arc_radian / 2, pad_n_rays + 1)
        pad_radians = (radian_edges[1:] + radian_edges[:-1]) / 2.
    else:
        pad_radians = np.linspace(pad_arc_radian / 2, -pad_arc_radian / 2, pad_n_rays)

    return pad_radians


def transmission_backproject(ray_geom, sinogram, image_shape, extent, step_size=1e-3, rays=None):
    def backproject_fan(ray_geom, sinogram, image_shape, extent):

        def pixel_center_radians(ray_geom, radian, image_shape):
            # calculate radian value at each pixel
            pixel_radians = np.zeros((image_shape[1] * image_shape[0]), dtype=np.double)
            pixel_radians = pixel_radians % np.pi * 2

            dx = (extent[1] - extent[0]) / image_shape[1]
            dy = (extent[3] - extent[2]) / image_shape[0]

            for j in range(image_shape[0]):
                for i in range(image_shape[1]):
                    center_x = extent[0] + (i + 0.5) * dx
                    center_y = extent[2] + (j + 0.5) * dy

                    source_x = -ray_geom['radius'] * np.cos(radian)
                    source_y = -ray_geom['radius'] * np.sin(radian)

                    value = np.arctan2(-source_y, -source_x) - np.arctan2(center_y - source_y, center_x - source_x)

                    pixel_radians[i + j * image_shape[1]] = -value

            pixel_radians += np.pi * 2
            pixel_radians = pixel_radians % (np.pi * 2)
            return pixel_radians

        backprojection = np.zeros(image_shape, dtype=np.double)
        radians = padded_radians(ray_geom)
        interp_radians = np.copy(radians) + (np.pi * 2)
        interp_radians = interp_radians % (np.pi * 2)

        for i, radian in enumerate(ray_geom['radians']):
            pixel_radians = pixel_center_radians(ray_geom, radian, image_shape)
            interp_sinogram = np.zeros(radians.shape[0])
            interp_sinogram[1:-1] = sinogram[i]

            backprojection += np.interp(pixel_radians, interp_radians, interp_sinogram, period=2*np.pi).reshape(image_shape)

        # return pixel_radians.reshape(image_shape)
        return backprojection

    def backproject_parallel(ray_geom, sinogram, image_shape, extent):

        def pixel_center_ss(ray_geom, radian, image_shape):
            # calculate radian value at each pixel
            pixel_ss = np.zeros((image_shape[1] * image_shape[0]), dtype=np.double)

            dx = (extent[1] - extent[0]) / image_shape[1]
            dy = (extent[3] - extent[2]) / image_shape[0]

            for j in range(image_shape[0]):
                for i in range(image_shape[1]):
                    center_x = extent[0] + (i + 0.5) * dx
                    center_y = extent[2] + (j + 0.5) * dy

                    ss = center_x * np.cos(radian) + center_y * np.sin(radian)

                    pixel_ss[i + j * image_shape[1]] = ss

            return pixel_ss

        backprojection = np.zeros(image_shape, dtype=np.double)

        for i, radian in enumerate(ray_geom['radians']):
            pixel_ss = pixel_center_ss(ray_geom, radian, image_shape)
            backprojection += np.interp(pixel_ss, ray_geom['ss'], sinogram[i], left=0, right=0).reshape(image_shape)

        return backprojection

    if ray_geom['type'] == 'fan':
        return backproject_fan(ray_geom, sinogram, image_shape, extent)
    elif ray_geom['type'] == 'parallel':
        return backproject_parallel(ray_geom, sinogram, image_shape, extent)


def c_transmission_backproject_fan(ray_geom, sinogram, image_shape, extent, step_size=1e-3, rays=None):
    backprojection = np.zeros(image_shape, dtype=np.double)
    radians = padded_radians(ray_geom)
    interp_sinogram = np.zeros(radians.shape[0])

    period = 2 * np.pi

    radians = radians % period
    asort_radian = np.argsort(radians)
    radians = radians[asort_radian]
    radians = np.concatenate((radians[-1:]-period, radians, radians[0:1]+period))

    for i, radian in enumerate(ray_geom['radians']):
        interp_sinogram[1:-1] = sinogram[i]
        interp_sinogram = interp_sinogram[asort_radian]
        interp_sinogram2 = np.concatenate((interp_sinogram[-1:], interp_sinogram, interp_sinogram[0:1]))

        source_x = -ray_geom['radius'] * np.cos(radian)
        source_y = -ray_geom['radius'] * np.sin(radian)

        c_backproject_fan(radians, interp_sinogram2, source_x, source_y, extent[0], extent[1], extent[2], extent[3], backprojection)

    return backprojection


def c_transmission_backproject_parallel(ray_geom, sinogram, image_shape, extent, step_size=1e-3, rays=None):
    backprojection = np.zeros(image_shape, dtype=np.double)

    for i, radian in enumerate(ray_geom['radians']):
        c_backproject_parallel(radian, ray_geom['ss'], sinogram[i], extent[0], extent[1], extent[2], extent[3], backprojection)

    return backprojection

# Fission Algorithms
def fission_project(rays, k, mu_image, mu_f_image, p_image, extent, step_size=1e-3):
    pass


def fission_backproject(rays, k, mu_image, mu_f_image, p_image, extent, step_size=1e-3):
    pass


def kaczmarz_reconstruction(n_steps, ray_geom, image_shape, image_extent, data_sinogram, step_size):
    rays = generate_rays(ray_geom)
    flat_rays = rays.reshape(-1, 4)
    indices = np.random.permutation(flat_rays.shape[0])
    m_old = np.zeros(image_shape, dtype=np.double)

    for ii in range(n_steps):
        i = ii % flat_rays.shape[0]
        iindex = indices[i]

        numerator = raytrace.raytrace_bilinear(flat_rays[iindex], m_old, extent=mu_im.extent, step_size=step_size)
        numerator -= data_sinogram.item(iindex)

        G_row = np.zeros(image_shape, dtype=np.double)
        raytrace.raytrace_backproject(flat_rays[iindex], 1, G_row, extent=image_extent, step_size=step_size)

        denominator = np.sqrt(np.sum(G_row.flatten() * G_row.flatten()))
        value = numerator / denominator

        m_new = np.zeros(mu_im.data.shape, dtype=np.double)
        raytrace.raytrace_backproject(flat_rays[iindex], value, m_new, extent=mu_im.extent, step_size=step_size)

        m_new = m_old - m_new
        m_old[:] = m_new

    return m_new


def CGLS_reconstruction(n_steps, ray_geom, image_shape, image_extent, data_sinogram, step_size):
    x = np.zeros(image_shape, dtype=np.double)
    r = data_sinogram - transmission_project(ray_geom, x, image_extent, step_size)
    d = transmission_backproject(ray_geom, r, image_shape, image_extent, step_size)

    AT_r = transmission_backproject(ray_geom, r, image_shape, image_extent, step_size)
    AT_rprevnorm = np.linalg.norm(AT_r)
    A_d = transmission_project(ray_geom, d, image_extent, step_size)

    for k in range(n_steps):
        print(k)
        alpha = (AT_rprevnorm ** 2) / (np.linalg.norm(A_d) ** 2)

        x = x + alpha * d
        r = r - alpha * A_d

        AT_r = transmission_backproject(ray_geom, r, image_shape, image_extent, step_size)

        beta = (np.linalg.norm(AT_r) ** 2) / (AT_rprevnorm ** 2)
        d = AT_r + beta * d

        A_d = transmission_project(ray_geom, d, image_extent, step_size)
        AT_rprevnorm = np.linalg.norm(AT_r)

    return x

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

    plt.figure()
    sino_extent = [-detector_arc_angle/2, detector_arc_angle/2, angles[0], angles[-1]]
    plt.imshow(sino, extent=sino_extent, aspect='auto', interpolation='nearest')

    plt.figure()
    plt.imshow(backprojection, extent=mu_im.extent, origin='lower')

    # kaczmarz stuff
    def kaczmarz_reconstruction(n_steps, rays, image_shape, image_extent, data_sinogram, step_size):
        flat_rays = rays.reshape(-1, 4)
        indices = np.random.permutation(flat_rays.shape[0])
        m_old = np.zeros(image_shape, dtype=np.double)

        for ii in range(n_steps):
            i = ii % flat_rays.shape[0]
            iindex = indices[i]

            numerator = raytrace.raytrace_bilinear(flat_rays[iindex], m_old, extent=mu_im.extent, step_size=step_size)
            numerator -= data_sinogram.item(iindex)

            G_row = np.zeros(image_shape, dtype=np.double)
            raytrace.raytrace_backproject(flat_rays[iindex], 1, G_row, extent=image_extent, step_size=step_size)

            denominator = np.sqrt(np.sum(G_row.flatten() * G_row.flatten()))
            value = numerator / denominator

            m_new = np.zeros(mu_im.data.shape, dtype=np.double)
            raytrace.raytrace_backproject(flat_rays[iindex], value, m_new, extent=mu_im.extent, step_size=step_size)

            m_new = m_old - m_new
            m_old[:] = m_new

        return m_new

    mu_kaczmarz = kaczmarz_reconstruction(40000 * 5, rays, mu_im.data.shape, mu_im.extent, sino, step_size)


    plt.figure()
    plt.imshow(mu_kaczmarz, extent=mu_im.extent, origin='lower')

    plt.show()
    """
    # Test Fission Sinogram
    """
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
    """
    # Test CGLS
    # """
    import cProfile
    # ray_geom = {'type': 'fan', 'radius': 40, 'detector_arc_radian': 40 / 180. * np.pi, 'n_rays': 200,
    #             'radians': np.linspace(0., 2 * np.pi, 100), 'midpoint': True}
    ray_geom = {'type': 'parallel', 'ss': np.linspace(-20, 20, 100), 'radians': np.linspace(0, np.pi, 100), 'length': 40}
    step_size = 0.00137

    mu_im, mu_f_im, p_im = assemblies.shielded_true_images()
    mu_im.data[:] = 0
    mu_im.data[20:50, 20:27] = 1

    rays = generate_rays(ray_geom)
    #

    pr = cProfile.Profile()
    pr.enable()

    sino = transmission_project(ray_geom, mu_im.data, mu_im.extent, step_size=step_size)
    # backprojection = transmission_backproject(ray_geom, sino, mu_im.data.shape, mu_im.extent, step_size=step_size) / len(ray_geom['radians'])
    backprojection = c_transmission_backproject_parallel(ray_geom, sino, mu_im.data.shape, mu_im.extent, step_size=step_size) / len(ray_geom['radians'])
    # backprojection = c_transmission_backproject_fan(ray_geom, sino, mu_im.data.shape, mu_im.extent, step_size=step_size) / len(ray_geom['radians'])
    pr.disable()
    pr.print_stats(sort='time')
    # print(sino)

    # mu_recon = CGLS_reconstruction(5, ray_geom, mu_im.data.shape, mu_im.extent, sino, step_size)
    # mu_recon = kaczmarz_reconstruction(40000 * 5, ray_geom, mu_im.data.shape, mu_im.extent, sino, step_size)

    plt.figure()
    plt.imshow(mu_im.data, extent=mu_im.extent, origin='lower')
    # draw_rays(rays[0])

    plt.figure()
    # sino_extent = [-ray_geom['detector_arc_radian'] / 2, ray_geom['detector_arc_radian'] / 2, 0, 2 * np.pi]
    sino_extent = [ray_geom['ss'][0], ray_geom['ss'][-1], ray_geom['radians'][0], ray_geom['radians'][-1]]
    plt.imshow(sino, extent=sino_extent, origin='lower', aspect='auto')
    #
    plt.figure()
    plt.imshow(backprojection, extent=mu_im.extent, origin='lower')

    # plt.figure()
    # plt.imshow(mu_recon, extent=mu_im.extent, origin='lower')

    plt.show()
    # """
    # test 1d interp
    """

    xs = np.array([-1, 0, 0.1, 2, 3], dtype=np.double)
    rxs = np.copy(xs[::-1])
    ys = xs ** 3 - 2 * xs ** 2 + 1

    plt.figure()
    plt.scatter(rxs, ys)

    xnew = np.linspace(-2, 4, 400, dtype=np.double)
    ynew = raytrace.interp(rxs, ys, xnew)

    # plt.figure()
    plt.plot(xnew, ynew)
    plt.scatter(xnew, ynew, marker='.')

    plt.show()

    """