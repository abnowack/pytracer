"""
TODO
 - [ ] Get Fission code working
    - [ ] Display singles / doubles sinograms and backprojections
 - [ ] Precalc all the detector probability stuff
 - [ ] Use precalc if available
 - [ ] Calculate changing p
 - [ ] Include changing p calculations
"""

import numpy as np
import matplotlib.pyplot as plt
import tomo
import assemblies


def ANGLE(x): return x / np.pi * 180.


def RADIAN(x): return x / 180. * np.pi


def draw_rays(rays, draw_option='b-', arrow=False):
    for ray in rays:
        dx = (ray[2] - ray[0]) / 2.
        dy = (ray[3] - ray[1]) / 2.
        if dx != 0 and dy != 0:
            plt.plot([ray[0], ray[2]], [ray[1], ray[3]], draw_option, lw=1)
            if arrow:
                plt.arrow(ray[0], ray[1], dx, dy,
                          fc='b', ec='b', head_width=0.5)


def draw_extent(extent, draw_option='b-'):
    draw_extent_xs = [extent[0], extent[1], extent[1], extent[0], extent[0]]
    draw_extent_ys = [extent[2], extent[2], extent[3], extent[3], extent[2]]

    plt.plot(draw_extent_xs, draw_extent_ys, draw_option)


def plot_sinogram(data, type, geo_angles, other_coord):
    plt.imshow(data, extent=[other_coord[-1], other_coord[0], geo_angles[-1], geo_angles[0]],
               origin='lower', aspect='auto')
    if type == 'fan':
        plt.xlabel(r'Fan Angle $\phi$')
        plt.ylabel(r'Geometry Angle $\theta$')
    elif type == 'parallel':
        plt.xlabel(r'Parallel Coord $r$')
        plt.ylabel(r'Geometry Angle $\theta$')


def draw_detector(points, draw_option=''):
    for i in range(points.shape[0]):
        plt.plot(points[i, (0, 2)], points[i, (1, 3)], lw=2)

    plt.scatter(points[:, (0, 2)], points[:, (1, 3)], color='k', s=8)


def plot_image(image, ex):
    plt.imshow(image, extent=ex, origin='lower', aspect='auto')


def CGLS_reconstruction(n_steps, image_shape, sinogram, A, AT):
    x = np.zeros(image_shape, dtype=np.double)
    r = sinogram - A(x)
    d = AT(r)

    AT_r = AT(r)
    AT_rprevnorm = np.linalg.norm(AT_r)
    A_d = A(d)

    for k in range(n_steps):
        print(k)
        alpha = (AT_rprevnorm ** 2) / (np.linalg.norm(A_d) ** 2)

        x = x + alpha * d
        r = r - alpha * A_d

        AT_r = AT(r)

        beta = (np.linalg.norm(AT_r) ** 2) / (AT_rprevnorm ** 2)
        d = AT_r + beta * d

        A_d = A(d)
        AT_rprevnorm = np.linalg.norm(AT_r)

    return x


def test_ray_crop():
    extent = np.array([-12, 12, -8, 8], dtype=np.double)

    ray_extent = [extent[0] - 2, extent[1] + 2, extent[2] - 2, extent[3] + 2]

    n_rays = 500

    rays = np.random.rand(n_rays, 4)
    rays[:, (0, 2)] *= (ray_extent[1] - ray_extent[0])
    rays[:, (0, 2)] -= (ray_extent[1] - ray_extent[0]) / 2
    rays[:, (1, 3)] *= (ray_extent[3] - ray_extent[2])
    rays[:, (1, 3)] -= (ray_extent[3] - ray_extent[2]) / 2

    plt.figure()
    draw_extent(extent, 'r-')

    crop_rays = tomo.ray_box_crop(rays, extent)
    draw_rays(crop_rays, 'g-')

    plt.axes().set_aspect('equal', 'datalim')
    plt.show()


def test_ray_geometry(type='parallel', theta=0., other_coord_n=50):
    extent = np.array([-12, 12, -8, 8], dtype=np.double)

    if type == 'parallel':
        length = 40
        r = np.linspace(-20, 20, other_coord_n)

        rays = tomo.parallel_ray(theta, r, length)
    if type == 'fan':
        radius = 40
        phi = np.linspace(-np.pi / 8, np.pi / 8, other_coord_n)

        rays = tomo.fan_ray(theta, phi, radius)

    crop_rays = tomo.ray_box_crop(rays, extent)

    plt.figure()
    draw_extent(extent)
    draw_rays(rays, arrow=True)
    plt.axes().set_aspect('equal', 'datalim')

    plt.figure()
    draw_extent(extent)
    draw_rays(crop_rays, arrow=True)

    plt.axes().set_aspect('equal', 'datalim')
    plt.show()


def test_detector_geometry(type='parallel', theta=0., n_detectors=10):
    if type == 'parallel':
        dr = 40.
        l = 40.

        detector_points = tomo.parallel_detector(n_detectors, theta, dr, l)
    elif type == 'fan':
        dphi = np.pi / 4
        radius = 40.

        detector_points = tomo.fan_detector(n_detectors, theta, dphi, radius)

    plt.figure()
    extent = np.array([-12, 12, -8, 8], dtype=np.double)
    draw_extent(extent)
    draw_detector(detector_points)

    plt.axes().set_aspect('equal', 'datalim')
    plt.show()


def test_bilinear():
    from scipy import interpolate

    image = np.zeros((7, 7), dtype=np.double)
    extent = np.array([-3.5, 3.5, -3.5, 3.5])

    image[5, 2] = 1
    image[5, 4] = 1

    image[3, 1] = 1
    image[2, 2] = 1
    image[2, 3] = 1
    image[2, 4] = 1
    image[3, 5] = 1

    image[0, 1] = 1

    plt.figure()
    plt.imshow(image, extent=extent, origin='lower')

    xs = np.linspace(-3, 3, 7)
    ys = np.linspace(-3, 3, 7)
    f = interpolate.interp2d(xs, ys, image, kind='linear')

    plt.figure()
    nx, ny = 500, 500
    xs = np.linspace(extent[0], extent[1], nx)
    ys = np.linspace(extent[2], extent[3], ny)
    zs = f(xs, ys)
    plt.imshow(zs, extent=extent, origin='lower')

    plt.figure()
    xs_, ys_ = np.meshgrid(xs, ys)
    xs_ = xs_.flatten()
    ys_ = ys_.flatten()
    my_zs = tomo.bilinear_interpolate(xs_, ys_, image, extent)
    print(my_zs)
    plt.imshow(my_zs.reshape(
        ys.shape[0], xs.shape[0]), extent=extent, origin='lower')

    plt.show()


def test_forward_project(type='parallel', theta_n=100, other_coord_n=100):

    theta = np.linspace(0., 2 * np.pi, theta_n, endpoint=False)
    step_size = 0.01

    mu_im, mu_f_im, p_im = assemblies.ut_logo()

    plt.figure()
    ex = list(mu_im.extent)
    plt.imshow(mu_im.data, extent=ex, origin='lower', aspect='auto')

    if type == 'parallel':
        length = 40
        r = np.linspace(-20, 20, other_coord_n)

        rays = tomo.parallel_ray(theta, r, length)

    if type == 'fan':
        radius = 40
        phi = np.linspace(-np.pi / 8, np.pi / 8, other_coord_n)

        rays = tomo.fan_ray(theta, phi, radius)

    sinogram = tomo.forward_project(rays, mu_im.data, mu_im.extent, step_size)

    plt.figure()

    if type == 'parallel':
        plot_sinogram(sinogram, 'parallel', theta, r)

    if type == 'fan':
        plot_sinogram(sinogram, 'fan', theta, phi)

    plt.show()


def test_back_project(type='parallel', theta=0., other_coord_n=100):

    step_size = 0.01

    mu_im, mu_f_im, p_im = assemblies.ut_logo()
    mu_im.data[:] = 0
    mu_im.data[20:50, 20:27] = 1

    nx, ny = mu_im.data.shape[1], mu_im.data.shape[0]
    extent = mu_im.extent

    plt.figure()
    plt.imshow(mu_im.data, extent=mu_im.extent, origin='lower', aspect='auto')

    if type == 'parallel':
        length = 40
        r = np.linspace(-20, 20, other_coord_n)

        rays = tomo.parallel_ray(theta, r, length)
        projection = tomo.forward_project(
            rays, mu_im.data, mu_im.extent, step_size)

        back_projection = tomo.back_project_parallel(
            theta, r, projection, nx, ny, extent)

    elif type == 'fan':
        radius = 40
        phi = np.linspace(-np.pi / 8, np.pi / 8, other_coord_n)

        rays = tomo.fan_ray(theta, phi, radius)
        projection = tomo.forward_project(
            rays, mu_im.data, mu_im.extent, step_size)

        back_projection = tomo.back_project_fan(
            theta, phi, radius, projection, nx, ny, extent)

    plt.figure()
    plot_image(back_projection, mu_im.extent)

    plt.show()


def test_cgls(steps_n, type='parallel', theta_n=100, other_coord_n=100):
    theta = np.linspace(0., 2 * np.pi, theta_n)
    step_size = 0.01

    mu_im, mu_f_im, p_im = assemblies.ut_logo()
    # mu_im.data[:] = 0
    # mu_im.data[20:50, 20:27] = 1

    nx, ny = mu_im.data.shape[1], mu_im.data.shape[0]
    extent = mu_im.extent

    if type == 'parallel':
        length = 40
        r = np.linspace(-20, 20, other_coord_n)

        rays = tomo.parallel_ray(theta, r, length)

        def A(x): return tomo.forward_project(rays, x, mu_im.extent, step_size)
        def AT(x): return tomo.back_project_parallel(
            theta, r, x, nx, ny, extent)

    if type == 'fan':
        radius = 40
        phi = np.linspace(-np.pi / 8, np.pi / 8, other_coord_n)

        rays = tomo.fan_ray(theta, phi, radius)

        def A(x): return tomo.forward_project(rays, x, mu_im.extent, step_size)
        def AT(x): return tomo.back_project_fan(
            theta, phi, radius, x, nx, ny, extent)

    sinogram = A(mu_im.data)
    backproject = AT(sinogram)

    cgls = CGLS_reconstruction(steps_n, mu_im.data.shape, sinogram, A, AT)

    plt.figure()
    plot_image(cgls, ex=mu_im.extent)

    plt.show()


"""
def test_detector_probability(type='parallel', geo_angle=0, n_detectors=20):
    step_size = 0.01

    mu_im, mu_f_im, p_im = assemblies.shielded_true_images()

    if type == 'parallel':
        width = 40
        radius = 40

        detector_points = tomo.parallel_detector(
            geo_angle, n_detectors, width, radius)

    if type == 'fan':
        detector_angle = np.pi / 4
        radius = 40

        detector_points = tomo.fan_detector(
            geo_angle, n_detectors, detector_angle, radius)

    detector_prob = tomo.precalculate_detector_probability(
        mu_im.data, mu_im.extent, detector_points, step_size)

    plt.figure()
    plot_image(detector_prob, mu_im.extent)

    plt.show()


def test_fission_project(type='parallel', k=1, geo_angle=0, other_coord_n=100, n_detectors=20, precalc=False):
    step_size = 0.01

    mu_im, mu_f_im, p_im = assemblies.shielded_true_images()

    nu_u235_induced = \
        np.array([0.0237898, 0.1555525, 0.3216515, 0.3150433,
                  0.1444732, 0.0356013, 0.0034339, 0.0004546])

    plt.figure()
    plot_image(mu_im)

    if type == 'parallel':
        length = 40
        parallel_coords = np.linspace(-20, 20, other_coord_n)

        width = 40
        radius = 40
        detector_points = tomo.parallel_detector(
            geo_angle, n_detectors, width, radius)

        draw_detector(detector_points)

        if not precalc:
            sinogram = tomo.fission_forward_project_parallel(
                geo_angle, parallel_coords, length, k,
                mu_im.data, mu_f_im.data, p_im.data, mu_im.extent,
                detector_points, nu_u235_induced, step_size)
        else:
            detector_prob = tomo.precalculate_detector_probability(
                mu_im.data, mu_im.extent, detector_points, step_size)

            sinogram = tomo.fission_precalc_forward_project_parallel(
                geo_angle, parallel_coords, length, k, mu_im.data, mu_f_im.data, p_im.data,
                detector_prob, mu_im.extent, nu_u235_induced, step_size)

        plt.figure()
        plt.plot(sinogram)

    if type == 'fan':
        radius = 40
        detector_angle = np.pi / 4

        fan_angles = np.linspace(- detector_angle / 2,
                                 detector_angle / 2, other_coord_n)

        detector_points = tomo.fan_detector(
            geo_angle, n_detectors, detector_angle, radius)

        draw_detector(detector_points)

        if not precalc:
            sinogram = tomo.fission_forward_project_fan(
                geo_angle, parallel_coords, length, k,
                mu_im.data, mu_f_im.data, p_im.data, mu_im.extent,
                detector_points, nu_u235_induced, step_size)

    plt.show()
"""

if __name__ == '__main__':
    # test_ray_crop()
    # test_ray_geometry('fan', RADIAN(301))
    # test_detector_geometry('parallel', RADIAN(301))
    # test_bilinear()
    # test_forward_project('fan')
    # test_back_project('fan', theta=np.linspace(
    #    RADIAN(0), RADIAN(360), 100, endpoint=False))
    test_cgls(5, 'parallel')
    # test_detector_probability('fan')
    # test_fission_project(k=3, other_coord_n=100, precalc=True)
