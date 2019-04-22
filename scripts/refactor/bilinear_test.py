import matplotlib.pyplot as plt
import numpy as np


def bilinear_interp_samp(x, y, pixels, extent):
    """
    NOTE: ASSUMES PIXELS IS ZERO PADDED
    a ---- b
    | x    |
    |      |
    c ---- d
    """
    delx = (extent[1] - extent[0]) / pixels.shape[1]
    dely = (extent[3] - extent[2]) / pixels.shape[0]

    if x < (extent[0] + delx / 2.) or x >= (extent[1] - delx / 2.):
        return 0
    if y < (extent[2] + dely / 2.) or y >= (extent[3] - dely / 2.):
        return 0

    # get index of lower left corner
    i1 = int(np.floor((x - extent[0] - delx / 2.) / (extent[1] - extent[0] - delx) * (pixels.shape[1] - 1)))
    j1 = int(np.floor((y - extent[2] - dely / 2.) / (extent[3] - extent[2] - dely) * (pixels.shape[0] - 1)))
    i2 = i1 + 1
    j2 = j1 + 1

    x1 = extent[0] + delx / 2. + i1 * delx
    y1 = extent[2] + dely / 2. + j1 * dely

    t = (x - x1) / delx
    u = (y - y1) / dely

    interp = (1 - t) * (1 - u) * pixels[j1, i1] + \
        t * (1 - u) * pixels[j1, i2] + \
        t * u * pixels[j2, i2] + \
        (1 - t) * u * pixels[j2, i1]

    return interp / (delx * dely)


def bilinear(image, extent, xs, ys):
    zs = np.zeros((xs.shape[0], ys.shape[0]))
    for i in range(xs.shape[0]):
        for j in range(ys.shape[0]):
            zs[j, i] = bilinear_interp_samp(xs[i], ys[j], image, extent)

    return zs


if __name__ == '__main__':
    from scipy import interpolate

    image = np.zeros((7, 7), dtype=np.double)
    extent = [-3.5, 3.5, -3.5, 3.5]

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
    my_zs = tomo.bilinear_interpolate(xs, ys, image, extent)
    plt.imshow(my_zs, extent=extent, origin='lower')


    plt.show()