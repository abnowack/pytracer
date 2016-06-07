import numpy as np
import sys

from intersect_module import intersecting_segments_c as isc
from geometries import build_shielded_geometry

if __name__ == "__main__":
    sim = build_shielded_geometry()

    segs = sim.geometry.mesh.segments


    print isc(4, segs)
    print segs[4]

