import numpy as np

from . import transmission as trans
from . import geometry as geo
from . import fission


def transmission_scan(flat_geom, start, end):
    absorb = np.zeros((start.shape[:-1]), dtype=np.double)
    flat_start = start.reshape(-1, start.shape[-1])
    flat_end = end.reshape(-1, end.shape[-1])

    trans.absorbances(flat_start, flat_end, flat_geom.segments, flat_geom.absorbance, 0, absorbance_cache=absorb.ravel())

    return absorb


# TODO Cython
def fission_scan(source, neutron_paths, detector_points, flat_geom, k, matrix, p_range):
    probs = np.zeros((np.size(source, 0), np.size(neutron_paths, 0)), dtype=np.double)
    for i in range(np.size(source, 0)):
        print(f'\r  {i} / {np.size(source, 0)-1}', end='', flush=True)
        detector_segments = geo.convert_points_to_segments(detector_points[:, i])
        for j in range(np.size(neutron_paths, 0)):
            probs[i, j] = fission.probability_path_neutron(source[i], neutron_paths[j, i], flat_geom, detector_segments,
                                                           k, matrix, p_range)
    print()

    return probs
