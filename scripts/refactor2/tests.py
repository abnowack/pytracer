import numpy as np
import matplotlib.pyplot as plt
import tomo

def test_bilinear():
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
    my_zs = bilinear(image, extent, xs, ys)
    plt.imshow(my_zs, extent=extent, origin='lower')

    plt.show()

input = np.ones((5, 5), dtype=np.double)
outs = cos_doubles.cos_doubles_func2D(input)

print(__file__)
print(os.getcwd())

print(input)
print()
print(outs)
