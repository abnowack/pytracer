"""
Create all of the assemblies of objects here
"""


from math import pi, floor
import numpy as np
from PIL import Image, ImageDraw

from utils import Data


def cartesian_to_image(x, y, extent, nx, ny):
    i = floor((x - extent[0]) / (extent[1] - extent[0]) * nx)
    j = floor((y - extent[2]) / (extent[3] - extent[2]) * ny)
    return i, j


def shielded_true_images():
    extent = [-12, 12, -8, 8]
    delta = 0.1
    nx = int((extent[1] - extent[0]) / delta)
    ny = int((extent[3] - extent[2]) / delta)

    u235 = 0.2
    steel = 0.15
    poly = 0.3

    origin = -9 + 3.8 + 0.3
    outer_radius = 3.8
    inner_radius = 2.8

    trans_im = Image.new('F', (nx, ny), color=0)
    draw = ImageDraw.Draw(trans_im)
    draw.rectangle([cartesian_to_image(-10, -5, extent, nx, ny),
                    cartesian_to_image(10, 5, extent, nx, ny)], fill=steel)
    draw.rectangle([cartesian_to_image(-9, -4, extent, nx, ny),
                    cartesian_to_image(9, 4, extent, nx, ny)], fill=0)

    draw.ellipse([cartesian_to_image(origin - outer_radius, -outer_radius, extent, nx, ny),
                  cartesian_to_image(origin + outer_radius, outer_radius, extent, nx, ny)], fill=u235)
    draw.ellipse([cartesian_to_image(origin - inner_radius, -inner_radius, extent, nx, ny),
                  cartesian_to_image(origin + inner_radius, inner_radius, extent, nx, ny)], fill=0)

    draw.rectangle([cartesian_to_image(5, 3, extent, nx, ny),
                    cartesian_to_image(7, 1, extent, nx, ny)], fill=steel)
    draw.rectangle([cartesian_to_image(5, -3, extent, nx, ny),
                    cartesian_to_image(7, -1, extent, nx, ny)], fill=poly)
    del draw
    trans_arr = np.array(trans_im, dtype=np.double)

    fission_im = Image.new('F', (nx, ny), color=0)
    draw = ImageDraw.Draw(fission_im)

    draw.ellipse([cartesian_to_image(origin - outer_radius, -outer_radius, extent, nx, ny),
                  cartesian_to_image(origin + outer_radius, outer_radius, extent, nx, ny)], fill=0.1)
    draw.ellipse([cartesian_to_image(origin - inner_radius, -inner_radius, extent, nx, ny),
                  cartesian_to_image(origin + inner_radius, inner_radius, extent, nx, ny)], fill=0)
    del draw
    fission_arr = np.array(fission_im, dtype=np.double)

    p_im = Image.new('F', (nx, ny), color=0)
    draw = ImageDraw.Draw(p_im)
    draw.ellipse([cartesian_to_image(origin - outer_radius, -outer_radius, extent, nx, ny),
                  cartesian_to_image(origin + outer_radius, outer_radius, extent, nx, ny)], fill=1.0)
    draw.ellipse([cartesian_to_image(origin - inner_radius, -inner_radius, extent, nx, ny),
                  cartesian_to_image(origin + inner_radius, inner_radius, extent, nx, ny)], fill=0)
    del draw
    p_mask = np.array(p_im, dtype=np.double)

    xs = np.arange(extent[0], extent[1], delta) + delta / 0.5
    ys = np.arange(extent[2], extent[3], delta) + delta / 0.5
    xs -= origin + 0.1
    ys -= 0
    ring_center_radius = (outer_radius - inner_radius) / 2 + inner_radius
    xv, yv = np.meshgrid(xs, ys)
    radius = np.sqrt(xv**2 + yv[::-1]**2)
    p_arr = -0.5 * (radius - ring_center_radius)**2 + 0.3
    slope = -0.05 / (1.1*3.8)
    p_arr += slope * xv - 0.05

    p_arr[p_mask != 1] = 0

    return [Data(extent, trans_arr), Data(extent, fission_arr), Data(extent, p_arr)]