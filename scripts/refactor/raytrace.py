import numpy as np
import pyximport; pyximport.install(setup_args={'include_dirs': np.get_include()})
from raytrace_c import c_raytrace_fast, c_raytrace_fast_bulk

_cache = np.zeros((500000), dtype=np.double)


def raytrace_fast(line, extent, pixels):
    return c_raytrace_fast(line, extent[0], extent[1], extent[2], extent[3], pixels)


def raytrace_bulk_fast(lines, extent, pixels):
    c_raytrace_fast_bulk(lines, extent[0], extent[1], extent[2], extent[3], pixels, _cache)
    return np.copy(_cache[:np.size(lines, 0)])
