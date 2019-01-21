import numpy as np
import matplotlib.pyplot as plt


def draw_alpha(line, alpha, color='red'):
    point_x = line[0] + alpha * (line[2] - line[0])
    point_y = line[1] + alpha * (line[3] - line[1])

    plt.scatter(point_x, point_y, color=color)


def raytrace(line, extent, pixels):

    x1, y1, x2, y2 = line

    # 1. alpha min and max

    Nx, Ny = np.size(pixels, 0) + 1, np.size(pixels, 1) + 1

    XPlane_1, XPlane_N, YPlane_1, YPlane_N = extent

    if x2 != x1:
        alpha_x_1 = (XPlane_1 - x1) / (x2 - x1)
        alpha_x_N = (XPlane_N - x1) / (x2 - x1)
        min_alpha_x = min(alpha_x_1, alpha_x_N)
        max_alpha_x = max(alpha_x_1, alpha_x_N)
    else:
        min_alpha_x = 0
        max_alpha_x = 1

    if y2 != y1:
        alpha_y_1 = (YPlane_1 - y1) / (y2 - y1)
        alpha_y_N = (YPlane_N - y1) / (y2 - y1)
        min_alpha_y = min(alpha_y_1, alpha_y_N)
        max_alpha_y = max(alpha_y_1, alpha_y_N)
    else:
        min_alpha_y = 0
        max_alpha_y = 1

    alpha_min = max(0, min_alpha_x, min_alpha_y)
    alpha_max = min(1, max_alpha_x, max_alpha_y)

    # 2. calculate range of indexes, i,j,k min and max

    dx = (XPlane_N - XPlane_1) / np.size(pixels, 0)
    dy = (YPlane_N - YPlane_1) / np.size(pixels, 1)

    if (x2 - x1) >= 0:
        i_min = Nx - (XPlane_N - alpha_min * (x2 - x1) - x1) / dx
        i_max = 1 + (x1 + alpha_max * (x2 - x1) - XPlane_1) / dx
    else:
        i_min = Nx - (XPlane_N - alpha_max * (x2 - x1) - x1) / dx
        i_max = 1 + (x1 + alpha_min * (x2 - x1) - XPlane_1) / dx
    if (y2 - y1) >= 0:
        j_min = Ny - (YPlane_N - alpha_min * (y2 - y1) - y1) / dy
        j_max = 1 + (y1 + alpha_max * (y2 - y1) - YPlane_1) / dy
    else:
        j_min = Ny - (YPlane_N - alpha_max * (y2 - y1) - y1) / dy
        j_max = 1 + (y1 + alpha_min * (y2 - y1) - YPlane_1) / dy

    # 3. calculate parametric sets alpha x,y,z

    i_min, i_max = int(i_min), int(i_max)
    j_min, j_max = int(j_min), int(j_max)

    if (x2 - x1) < 0:
        alpha_x = [((XPlane_1 + (i - 1) * dx) - x1) / (x2 - x1) for i in range(i_min, i_max+1)]
    elif x2 != x1:
        alpha_x = [((XPlane_1 + (i - 1) * dx) - x1) / (x2 - x1) for i in range(i_max, i_min + 1)]
    else:
        alpha_x = []

    if (y2 - y1) < 0:
        alpha_y = [((YPlane_1 + (j - 1) * dy) - y1) / (y2 - y1) for j in range(j_min, j_max+1)]
    elif y2 != y1:
        alpha_y = [((YPlane_1 + (j - 1) * dy) - y1) / (y2 - y1) for j in range(j_max, j_min + 1)]
    else:
        alpha_y = []

    # 4. merge to form set alpha

    alphas = [alpha_min] + alpha_x + alpha_y + [alpha_max]
    alphas = sorted(alphas)
    alphas_sort = [alpha for n, alpha in enumerate(alphas) if alpha not in alphas[:n] and alpha >= alpha_min and alpha <= alpha_max]

    alphas_mid = [(alphas_sort[i] + alphas_sort[i-1]) / 2 for i in range(1, len(alphas_sort))]

    i_index = [int(1 + (x1 + alpha_mid * (x2 - x1) - XPlane_1) / dx) for alpha_mid in alphas_mid]
    j_index = [int(1 + (y1 + alpha_mid * (y2 - y1) - YPlane_1) / dy) for alpha_mid in alphas_mid]

    for (i, j) in zip(i_index, j_index):
        pixels[i-1, j-1] = 1

    draw_algorithm(line, extent, pixels)
    for alpha in alphas_mid:
        draw_alpha(line, alpha)

    # 5. calculate voxel lengths
    # 6. calculate voxel indices

def raytrace2(line, extent, pixels):
    from math import floor, ceil, sqrt

    Nx, Ny = np.size(pixels, 0) + 1, np.size(pixels, 1) + 1
    p1x, p1y, p2x, p2y = line
    bx, by = extent[0], extent[2]
    dx, dy = (extent[1] - extent[0]) / (Nx - 1), (extent[3] - extent[2]) / (Ny - 1)

    px = lambda a: p1x + a * (p2x - p1x)
    py = lambda a: p1y + a * (p2y - p1y)

    alpha_x = lambda i: ((bx + i * dx) - p1x) / (p2x - p1x)
    alpha_y = lambda j: ((by + j * dy) - p1y) / (p2y - p1y)

    alpha_xmin = min(alpha_x(0), alpha_x(Nx - 1))
    alpha_ymin = min(alpha_y(0), alpha_y(Ny - 1))

    alpha_xmax = max(alpha_x(0), alpha_x(Nx - 1))
    alpha_ymax = max(alpha_y(0), alpha_y(Ny - 1))

    # for a ray, remove the 0, 1 in min max calcs
    alpha_min = max(0, alpha_xmin, alpha_ymin)
    alpha_max = min(1, alpha_xmax, alpha_ymax)

    if alpha_min >= alpha_max:
        raise ArithmeticError

    phi_x = lambda a: (p1x + a * (p2x - p1x) - bx) / dx
    phi_y = lambda a: (p1y + a * (p2y - p1y) - by) / dy

    if p1x < p2x:
        if alpha_min == alpha_xmin:
            i_min = 1
        else:
            i_min = ceil(phi_x(alpha_min))

        if alpha_max == alpha_xmax:
            i_max = Nx - 1
        else:
            i_max = floor(phi_x(alpha_max))
    else:
        if alpha_min == alpha_xmin:
            i_max = Nx - 2
        else:
            i_max = floor(phi_x(alpha_min))

        if alpha_max == alpha_xmax:
            i_min = 0
        else:
            i_min = ceil(phi_x(alpha_max))

    if p1y < p2y:
        if alpha_min == alpha_ymin:
            j_min = 1
        else:
            j_min = ceil(phi_y(alpha_min))

        if alpha_max == alpha_ymax:
            j_max = Ny - 1
        else:
            j_max = floor(phi_y(alpha_max))
    else:
        if alpha_min == alpha_ymin:
            j_max = Ny - 2
        else:
            j_max = floor(phi_y(alpha_min))

        if alpha_max == alpha_ymax:
            j_min = 0
        else:
            j_min = ceil(phi_y(alpha_max))

    if p1x < p2x:
        alpha_x_arr = [alpha_x(i) for i in range(i_min, i_max + 1)]
    else:
        alpha_x_arr = [alpha_x(i) for i in range(i_max, i_min - 1, -1)]

    if p1y < p2y:
        alpha_y_arr = [alpha_y(j) for j in range(j_min, j_max + 1)]
    else:
        alpha_y_arr = [alpha_y(j) for j in range(j_max, j_min - 1, -1)]

    # for a ray, remove [alpha_max]
    alpha_arr = sorted([alpha_min] + alpha_x_arr + alpha_y_arr + [alpha_max])
    alpha_arr = [a for n, a in enumerate(alpha_arr) if a not in alpha_arr[:n]]

    pixel_i = [floor(phi_x((alpha_arr[m] + alpha_arr[m-1]) / 2)) for m in range(1, len(alpha_arr))]
    pixel_j = [floor(phi_y((alpha_arr[m] + alpha_arr[m-1]) / 2)) for m in range(1, len(alpha_arr))]

    print([(pi, pj) for (pi, pj) in zip(pixel_i, pixel_j)])

    for (pi, pj) in zip(pixel_i, pixel_j):
        pixels[pi, pj] = 1

    draw_algorithm(line, extent, pixels)
    for alpha in alpha_arr:
        draw_alpha(line, alpha)

    d_conv = sqrt((p2x - p2x)**2 + (p2y - p1y)**2)
    lengths = [(alpha_arr[m] - alpha_arr[m-1]) * d_conv for m in range(1, len(alpha_arr))]
    integral = sum(pixels[pixel_i, pixel_j] * lengths)
    print(integral, d_conv)


def raytrace3(line, extent, pixels):
    from math import floor, ceil, sqrt

    Nx, Ny = np.size(pixels, 0) + 1, np.size(pixels, 1) + 1
    p1x, p1y, p2x, p2y = line
    bx, by = extent[0], extent[2]
    dx, dy = (extent[1] - extent[0]) / (Nx - 1), (extent[3] - extent[2]) / (Ny - 1)

    px = lambda a: p1x + a * (p2x - p1x)
    py = lambda a: p1y + a * (p2y - p1y)

    alpha_x = lambda i: ((bx + i * dx) - p1x) / (p2x - p1x)
    alpha_y = lambda j: ((by + j * dy) - p1y) / (p2y - p1y)
    print(alpha_x(0), alpha_y(0))

    alpha_xmin = min(alpha_x(0), alpha_x(Nx - 1))
    alpha_ymin = min(alpha_y(0), alpha_y(Ny - 1))

    alpha_xmax = max(alpha_x(0), alpha_x(Nx - 1))
    alpha_ymax = max(alpha_y(0), alpha_y(Ny - 1))

    # for a ray, remove the 0, 1 in min max calcs
    alpha_min = max(alpha_xmin, alpha_ymin)
    alpha_max = min(alpha_xmax, alpha_ymax)

    print(alpha_min, alpha_max)
    print()

    if alpha_min >= alpha_max:
        raise ArithmeticError

    # different here
    phi_x = lambda a: (px(a) - bx) / dx
    phi_y = lambda a: (py(a) - by) / dy

    if p1x < p2x:
        if alpha_min == alpha_xmin:
            i_min = 1
        else:
            i_min = ceil(phi_x(alpha_min))

        if alpha_max == alpha_xmax:
            i_max = Nx - 1
        else:
            i_max = floor(phi_x(alpha_max))
    else:
        if alpha_min == alpha_xmin:
            i_max = Nx - 2
        else:
            i_max = floor(phi_x(alpha_min))

        if alpha_max == alpha_xmax:
            i_min = 0
        else:
            i_min = ceil(phi_x(alpha_max))

    if p1y < p2y:
        if alpha_min == alpha_ymin:
            j_min = 1
        else:
            j_min = ceil(phi_y(alpha_min))

        if alpha_max == alpha_ymax:
            j_max = Ny - 1
        else:
            j_max = floor(phi_y(alpha_max))
    else:
        if alpha_min == alpha_ymin:
            j_max = Ny - 2
        else:
            j_max = floor(phi_y(alpha_min))

        if alpha_max == alpha_ymax:
            j_min = 0
        else:
            j_min = ceil(phi_y(alpha_max))

    Np = (i_max - i_min + 1) + (j_max - j_min + 1)
    ax = alpha_x(0)
    ay = alpha_y(0)

    mid = (min(ax, ay) + alpha_min) / 2
    print(px(mid), py(mid))
    print(mid, phi_x(mid), phi_y(mid))
    i_ = floor(phi_x(mid))
    j_ = floor(phi_y(mid))
    print(i_, j_)

    d12 = 0
    a_c = alpha_min

    axu = dx / abs(p2x - p1x)
    ayu = dy / abs(p2y - p1y)

    draw_algorithm(line, extent, pixels)
    # draw_alpha(line, alpha_min)

    for k in range(Np):
        if ax < ay:
            d12 += (ax - a_c) * pixels[i_, j_]
            i_ += 1 if p1x < p2x else -1
            a_c = ax
            ax += axu
        else:
            d12 += (ay - a_c) * pixels[i_, j_]
            j_ += 1 if p1y < p2y else -1
            a_c = ay
            ay += ayu
        draw_alpha(line, a_c)
        # print(ax, ay)
        # print(i_, j_)
        # print()

    draw_alpha(line, alpha_x(0))
    draw_alpha(line, alpha_y(0))

    d_conv = sqrt((p2x - p2x) ** 2 + (p2y - p1y) ** 2)
    d12 *= d_conv

    print(d12, d_conv)


def raytrace4(line, extent, pixels):
    # Fixed issue, alphax[0] in Filip Jacob's paper means first alphax in siddon array, not alphax at zero.
    # need to modify for a segment contained within pixel space, neccessary for fission calcs

    from math import ceil, floor

    p1x, p1y, p2x, p2y = line
    bx, by, = extent[0], extent[2]
    Nx, Ny = np.size(pixels, 0) + 1, np.size(pixels, 1) + 1
    dx, dy = (extent[1] - extent[0]) / np.size(pixels, 0), (extent[3] - extent[2]) / np.size(pixels, 1)

    p12x = lambda a_: p1x + a_ * (p2x - p1x)
    p12y = lambda a_: p1y + a_ * (p2y - p1y)

    alphax = lambda i_: ((bx + i_ * dx) - p1x) / (p2x - p1x)
    alphay = lambda j_: ((by + j_ * dy) - p1y) / (p2y - p1y)

    alphaxmin = min(alphax(0), alphax(Nx - 1))
    alphaxmax = max(alphax(0), alphax(Nx - 1))

    alphaymin = min(alphay(0), alphay(Ny - 1))
    alphaymax = max(alphay(0), alphay(Ny - 1))

    # consider having two options, functions, etc
    # if for a line segment, include the 0 in max and 1 in min
    # if for an infinite line, remove
    alphamin = max(0, alphaxmin, alphaymin)
    alphamax = min(1, alphaxmax, alphaymax)

    phix = lambda a_: (p12x(a_) - bx) / dx

    if p1x < p2x:
        if alphamin == alphaxmin:
            imin = 1
        else:
            imin = ceil(phix(alphamin))

        if alphamax == alphaxmax:
            imax = Nx - 1
        else:
            imax = floor(phix(alphamax))

        alphax_ = alphax(imin)

    else:
        if alphamin == alphaxmin:
            imax = Nx - 2
        else:
            imax = floor(phix(alphamin))

        if alphamax == alphaxmax:
            imin = 0
        else:
            imin = ceil(phix(alphamax))

        alphax_ = alphax(imax)

    phiy = lambda a_: (p12y(a_) - by) / dy

    if p1y < p2y:
        if alphamin == alphaymin:
            jmin = 1
        else:
            jmin = ceil(phiy(alphamin))

        if alphamax == alphaymax:
            jmax = Ny - 1
        else:
            jmax = floor(phiy(alphamax))

        alphay_ = alphay(jmin)

    else:
        if alphamin == alphaymin:
            jmax = Ny - 2
        else:
            jmax = floor(phiy(alphamin))

        if alphamax == alphaymax:
            jmin = 0
        else:
            jmin = ceil(phiy(alphamax))

        alphay_ = alphay(jmax)

    Np = (imax - imin + 1) + (jmax - jmin + 1)

    alphamid = (min(alphax_, alphay_) + alphamin) / 2

    i = floor(phix(alphamid))
    j = floor(phiy(alphamid))

    print(i, j)

    alphaxu = dx / abs(p2x - p1x)
    alphayu = dy / abs(p2y - p1y)

    d12 = 0
    dconv = ((p2x - p1x)**2 + (p2y - p1y)**2)**0.5
    alphac = alphamin
    draw_alpha(line, alphac)

    for k in range(Np):
        if p1x < p2x:
            iu = 1
        else:
            iu = -1

        if p1y < p2y:
            ju = 1
        else:
            ju = -1

        pixels[i, j] = 1

        if alphax_ < alphay_:
            lij = (alphax_ - alphac) * dconv
            d12 = d12 + lij * 1
            i = i + iu
            alphac = alphax_
            alphax_ = alphax_ + alphaxu
        else:
            lij = (alphay_ - alphac) * dconv
            d12 = d12 + lij * 1
            j = j + ju
            alphac = alphay_
            alphay_ = alphay_ + alphayu

        draw_alpha(line, alphac)

    print(alphamax, alphamin)
    print(d12, dconv * (alphamax - alphamin))

    draw_algorithm(line, extent, pixels)

def draw_algorithm(line, extent, pixels):

    plt.imshow(pixels.T, extent=extent, origin='lower')

    plt.plot([line[0], line[2]], [line[1], line[3]], 'k-')

    # vertical lines
    for i in range(np.size(pixels, 0) + 1):
        x = extent[0] + (extent[1] - extent[0]) / np.size(pixels, 0) * i
        plt.plot([x, x], [extent[2], extent[3]], 'g')

    # horizontal lines
    for i in range(np.size(pixels, 1) + 1):
        y = extent[2] + (extent[3] - extent[2]) / np.size(pixels, 1) * i
        plt.plot([extent[0], extent[1]], [y, y], 'g')


if __name__ == '__main__':
    image = np.zeros((30, 30))
    extent = [-5, 5, -5, 5]
    line = [-4.34, -3.1, 2.11, 6.1]

    raytrace4(line, extent, image)

    plt.show()
