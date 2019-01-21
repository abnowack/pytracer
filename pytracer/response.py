import numpy as np

from . import geometry as geo
from . import transmission as transmission
from . import neutron_chain as chain
from . import fission


def transmission_grid_response(flat_geom, grid, start, end):
    unit_m = geo.Material('black', 1, 0, 0)
    vacuum = geo.Material('white', 0, 0, 0)

    flat_start = start.reshape(-1, start.shape[-1])
    flat_end = end.reshape(-1, end.shape[-1])

    response = np.zeros((grid.num_cells,) + start.shape[:-1])
    response_shape = response.shape
    response = response.reshape((response.shape[0], response.shape[1] * response.shape[2]))

    for i in range(grid.num_cells):
        cell_geom = [geo.Solid(geo.convert_points_to_segments(grid.cell(i), circular=True), unit_m, vacuum)]
        cell_flat = geo.flatten(cell_geom)

        transmission.absorbances(flat_start, flat_end, cell_flat.segments, cell_flat.absorbance,
                                 absorbance_cache=response[i, :])

    return response.reshape(response_shape)


# TODO Cython
def fission_grid_response(source, neutron_paths, detector_points, grid, flat_geom, k_, matrix, p_range, p_model):
    # unit_m used for finding fission segments, not in fission prob calculation
    unit_m = geo.Material('black', 1, 1, 0)
    vacuum = geo.Material('white', 0, 0, 0)

    response = np.zeros((grid.num_cells, np.size(source, 0), np.size(neutron_paths, 0)), dtype=np.double)

    for i in range(grid.num_cells):
        print(i, ' / ', grid.num_cells)
        cell_geom = [geo.Solid(geo.convert_points_to_segments(grid.cell(i), circular=True), unit_m, vacuum)]
        cell_flat = geo.flatten(cell_geom)

        nu_dist = chain.interpolate_p(matrix, p_model[i], p_range, method='linear', log_interpolate=False)

        for j in range(np.size(source, 0)):
            detector_segments = geo.convert_points_to_segments(detector_points[:, j])
            for k in range(np.size(neutron_paths, 0)):
                segment, val = fission.find_fission_segments(source[j], neutron_paths[k, j], cell_flat)
                if len(segment) != 0:
                    segment = segment[0]
                    response[i, j, k] = fission.probability_segment_neutron_grid_c(source[j], segment, flat_geom,
                                                                                   detector_segments, k_, nu_dist)

    return response
