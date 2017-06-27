"""
Create all of the assemblies of objects here
"""


from pytracer import geometry as geo
from math import pi


def foo(x, y):
    return 1


_circle_inner_radius = 2.9
_circle_outer_radius = 3.9
_circle_origin = [-9 + _circle_outer_radius + 0.1, 0.]

def shielded_assembly(fission=False):
    air = geo.Material('white', 0, 0, None)
    u235_metal = geo.Material('green', 0.2, 0.1, foo)
    poly = geo.Material('red', 0.3, 0, None)
    steel = geo.Material('orange', 0.18, 0, None)

    box = geo.create_hollow(geo.create_rectangle(20, 10), geo.create_rectangle(18, 8))

    hollow_circle = geo.create_hollow(geo.create_circle(_circle_outer_radius), geo.create_circle(_circle_inner_radius))
    # hollow_circle = geo.create_arc(3.9)
    geo.translate(hollow_circle, _circle_origin)

    small_box_1 = geo.create_rectangle(2, 2)
    geo.translate(small_box_1, [6, 2])

    small_box_2 = geo.create_rectangle(2, 2)
    geo.translate(small_box_2, [6, -2])

    solids = [geo.Solid(box, steel, air), geo.Solid(hollow_circle, u235_metal, air),
              geo.Solid(small_box_1, poly, air), geo.Solid(small_box_2, steel, air)]

    return solids


def ray_trace_test_assembly():
    air = geo.Material('white', 0, 0, None)
    steel = geo.Material('orange', 0.18, 0, None)

    box = geo.create_hollow(geo.create_rectangle(12, 12), geo.create_rectangle(10, 10))
    ring = geo.create_hollow(geo.create_arc(12), geo.create_arc(10))
    geo.rotate(box, pi / 4)

    solids = [geo.Solid(ring, steel, air)]

    return solids
