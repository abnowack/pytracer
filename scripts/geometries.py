from pytracer.mesh import create_hollow, create_rectangle, create_circle
from pytracer.solid import Solid
from pytracer.material import Material
from pytracer.simulation import Simulation

def build_shielded_geometry(fission=False):
    air = Material(0.001, color='white')
    u235_metal = Material(0.2, 0.1, color='green')
    poly = Material(0.3, color='red')
    steel = Material(0.18, color='orange')

    box = create_hollow(create_rectangle(20., 10.), create_rectangle(18., 8.))

    hollow_circle = create_hollow(create_circle(3.9), create_circle(2.9))
    hollow_circle.translate([-9 + 3.9 + 0.1, 0.])

    small_box_1 = create_rectangle(2., 2.)
    small_box_1.translate([6., 2.])

    small_box_2 = create_rectangle(2., 2.)
    small_box_2.translate([6., -2.])

    sim = Simulation(air)
    sim.add_aligned_source_detector(diameter=50., nbins=100, width=30.)
    sim.geometry.solids.append(Solid(box, steel, air))
    if fission:
        sim.geometry.solids.append((Solid(hollow_circle, u235_metal, air)))
    else:
        sim.geometry.solids.append(Solid(hollow_circle, steel, air))
    sim.geometry.solids.append(Solid(small_box_1, poly, air))
    sim.geometry.solids.append(Solid(small_box_2, steel, air))
    sim.geometry.flatten()

    return sim


def ray_trace_test_geometry():
    air = Material(0.0, color='white')
    steel = Material(1.0, color='red')

    box = create_hollow(create_rectangle(12., 12.), create_rectangle(10., 10.))
    ring = create_hollow(create_circle(12.), create_circle(10.))
    box.rotate(45.)

    sim = Simulation(air, 100, diameter=50.)
    sim.detector.width = 30.
    sim.geometry.solids.append(Solid(ring, steel, air))
    sim.geometry.flatten()

    return sim