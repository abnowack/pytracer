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


def draw_algorithm(extent, pixels, draw_lines=True):

    plt.imshow(pixels.T, extent=extent, origin='lower')

    if draw_lines:
        # vertical lines
        for i in range(np.size(pixels, 0) + 1):
            x = extent[0] + (extent[1] - extent[0]) / np.size(pixels, 0) * i
            plt.plot([x, x], [extent[2], extent[3]], 'g')

        # horizontal lines
        for i in range(np.size(pixels, 1) + 1):
            y = extent[2] + (extent[3] - extent[2]) / np.size(pixels, 1) * i
            plt.plot([extent[0], extent[1]], [y, y], 'g')


def draw_line(line, draw_option='c-'):
    plt.plot([line[0], line[2]], [line[1], line[3]], draw_option, lw=1)


def draw_rays(rays, draw_option='b-'):
    for ray in rays:
        plt.plot([ray[0], ray[2]], [ray[1], ray[3]], draw_option, lw=1)


def parallel_rays(minx, maxx, miny, maxy, n_rays):
    rays = np.zeros((n_rays, 4), dtype=np.double)
    rays[:, 0] = minx
    rays[:, 2] = maxx
    rays[:, 1] = np.linspace(miny, maxy, n_rays)
    rays[:, 3] = rays[:, 1]

    return rays


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


def fan_rays(radius, arc_angle, n_rays, midpoint=False):
    rays = np.zeros((n_rays, 4), dtype=np.double)
    rays[:, (0, 1)] = np.array([-radius / 2, 0])
    rays[:, (2, 3)] = arc_detector_points(-radius / 2, 0, radius, arc_angle, n_rays, midpoint)

    return rays


def sinogram(rays, image, extent, **options):
    return raytrace.raytrace_bulk_bilinear(rays, extent, image, **options)


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


def rotation_sinogram(rays, image, extent, angles, **options):

    result = np.zeros((angles.shape[0], rays.shape[0]), dtype=np.double)

    for i, angle in enumerate(angles):
        rotated_rays = rotate_rays(rays, angle)
        result[i] = sinogram(rotated_rays, image, extent, **options)

    return result


def transmission_pixel_sinogram(rays, image, angles, pixel_i, pixel_j, rays_downsample=5, response_image=None):
    if response_image is None:
        response_image = Data(image.extent, np.zeros(image.data.shape, dtype=np.double))
        response_image.data[pixel_i, pixel_j] = 1

    response = rotation_sinogram(rays, response_image, angles)
    response = response.reshape(response.shape[0], -1, rays_downsample).mean(axis=2)

    return response


def transmission_response_sinogram(rays, image, angles, rays_downsample=5):
    nx, ny = image.data.shape
    response = np.zeros((nx * ny, angles.shape[0], int(rays.shape[0] / rays_downsample)), dtype=np.double)

    response_image = Data(image.extent, np.zeros(image.data.shape, dtype=np.double))

    for i in range(nx * ny):
        ix = i % image.data.shape[0]
        iy = i // image.data.shape[0]
        print(i, nx * ny, ix, iy)
        response_image.data[ix, iy] = 1
        response[i] = transmission_pixel_sinogram(rays, image, angles, ix, iy, rays_downsample, response_image)
        response_image.data[ix, iy] = 0

    return response


def transmission_project(image, extent, angles, radius, detector_arcangle, n_rays=200, step_size=1e-3):
    projection_rays = fan_rays(radius*2, detector_arcangle, n_rays, midpoint=True)

    return rotation_sinogram(projection_rays, image, extent, angles, step_size=step_size)


def transmission_backproject(sino, image_shape, extent, angles, radius, detector_arcangle, n_rays=200, step_size=1e-3):
    projection_rays = fan_rays(radius * 2, detector_arcangle, n_rays, midpoint=True)

    backprojection = np.zeros(image_shape, dtype=np.double)
    for i in range(angles.shape[0]):
        backproject_rays = rotate_rays(projection_rays, angles[i])
        raytrace.raytrace_backproject_bulk(backproject_rays, sino[i], mu_im.extent, backprojection, step_size=0.05)

    return backprojection


# fission stuff
def binom(n, k):

    ans = 1

    if k > n - k:
        k = n - k

    for j in range(1, k+1):
        if n % j == 0:
            ans *= n / j
        elif ans % j == 0:
            ans = ans / j * n
        else:
            ans = (ans * n) / j

        n -= 1

    return ans


def pixel_prob_detect(pixel_i, pixel_j, detector_points, extent, mu_image):
    def solid_angle_line(sx, sy, x1, y1, x2, y2):
        a = (x1 - sx) ** 2 + (y1 - sy) ** 2
        b = (x2 - sx) ** 2 + (y2 - sy) ** 2
        c = (x1 - x2) ** 2 + (y1 - y2) ** 2

        num = a + b - c
        denom = 2 * np.sqrt(a) * np.sqrt(b)
        angle = np.arccos(np.abs(num / denom))
        if angle > np.pi / 2.:
            angle = np.pi - angle
        return angle

    pixel_x = pixel_i / (mu_image.shape[0]) * (extent[1] - extent[0]) + extent[0] + 0.5 * (extent[1] - extent[0]) / mu_image.shape[0]
    pixel_y = pixel_j / (mu_image.shape[1]) * (extent[3] - extent[2]) + extent[2] + 0.5 * (extent[3] - extent[2]) / mu_image.shape[1]

    prob_detect = 0

    for i in range(detector_points.shape[0] - 1):
        detector_center_x = (detector_points[i, 0] + detector_points[i+1, 0]) / 2.
        detector_center_y = (detector_points[i, 1] + detector_points[i+1, 1]) / 2.
        line = np.array([pixel_x, pixel_y, detector_center_x, detector_center_y], dtype=np.double)

        absorbance_out = raytrace.raytrace_fast(line, extent, mu_image)
        exit_prob = np.exp(-absorbance_out)

        solid_angle = solid_angle_line(pixel_x, pixel_y, detector_points[0, 0], detector_points[0, 1],
                                       detector_points[1, 0], detector_points[1, 1])
        prob_detect += solid_angle * exit_prob

    return prob_detect


def pixel_fission_response_sinogram(rays, mu_image, mu_f_image, p_image, angles):
    pass


def pixel_fission_response(source_x, source_y, pixel_i, pixel_j, extent, k, detector_points, mu_image, mu_f_image, p_image):

    pixel_x = pixel_i / (mu_image.shape[0]) * (extent[1] - extent[0]) + extent[0] + 0.5 * (extent[1] - extent[0]) / mu_image.shape[0]
    pixel_y = pixel_j / (mu_image.shape[1]) * (extent[3] - extent[2]) + extent[2] + 0.5 * (extent[3] - extent[2]) / mu_image.shape[1]

    line = np.array([source_x, source_y, pixel_x, pixel_y], dtype=np.double)
    attenuation = raytrace.raytrace_fast(line, extent, mu_image)

    prob_in = np.exp(-attenuation)
    prob_density_fission = mu_f_image[pixel_i, pixel_j]

    # prob_detect = pixel_prob_detect(pixel_i, pixel_j, detector_points, extent, mu_image)
    prob_detect = 0.2

    p_fission = p_image[pixel_i, pixel_j]
    # nu_dist = nu_distribution(p_fission)
    nu_dist = np.array([0.0237898, 0.1555525, 0.3216515, 0.3150433, 0.1444732, 0.0356013, 0.0034339, 0.0004546])

    prob_out = 0
    for i in range(nu_dist.shape[0]):
        prob_out += binom(i, k) * nu_dist[i] * (prob_detect ** k) * ((1 - prob_detect) ** (i - k))

    pixel_prob_density = prob_in * prob_out * prob_density_fission

    return pixel_prob_density


def fission_sinogram(source_rays, detector_points, extent, mu_image, mu_f_image, p_image, k=1, rays_downsample=5):
    fission_response = np.zeros((source_rays.shape[0]), dtype=np.double)
    for i in range(source_rays.shape[0]):
        line = source_rays[i]
        pixels, distances = raytrace.raytrace_fast_store(line, extent, mu_image)
        for j in range(pixels.shape[0]):
            fission_response[i] += pixel_fission_response(line[0], line[1], pixels[j, 0], pixels[j, 1], extent, k,
                                                          detector_points, mu_image, mu_f_image, p_image)

    return fission_response


def fission_rotation_sinogram(source_rays, detector_points, angles, extent, mu_image, mu_f_image, p_image, k=1):

    result = np.zeros((angles.shape[0], source_rays.shape[0]), dtype=np.double)

    for i, angle in enumerate(angles):
        rotated_rays = rotate_rays(source_rays, angle)
        rotated_detector_points = rotate_points(detector_points, angle)
        print(result.shape)
        result[i] = fission_sinogram(rotated_rays, rotated_detector_points, extent, mu_image, mu_f_image, p_image, k)

    return result


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
    radius = 40
    detector_arc_angle = 40
    n_rays=200
    step_size = 0.05
    angles = np.linspace(0, 360, 200)

    mu_im, mu_f_im, p_im = assemblies.shielded_true_images()

    plt.figure()
    plt.imshow(mu_im.data, extent=mu_im.extent, origin='lower')

    sino = transmission_project(mu_im.data, mu_im.extent, angles, radius, detector_arc_angle, n_rays, step_size=step_size)

    plt.figure()
    plt.imshow(sino, extent=[-detector_arc_angle/2, detector_arc_angle/2, angles[0], angles[-1]], aspect='auto', interpolation='nearest')

    backprojection = transmission_backproject(sino, mu_im.data.shape, mu_im.extent, angles, radius, detector_arc_angle, n_rays, step_size=1e-3)

    plt.figure()
    plt.imshow(backprojection, extent=mu_im.extent, origin='lower')

    plt.show()
