import sys
import matplotlib.pyplot as plt
import numpy as np
from scripts.assemblies import shielded_assembly
import pytracer.geometry as geo
import pytracer.transmission as transmission
import pytracer.algorithms as algorithms
import pytracer.fission as fission


def rebin(a, shape):
    sh = shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1]
    return a.reshape(sh).mean(-1).mean(1)


if __name__ == "__main__":
    assembly_solids = shielded_assembly()
    assembly_flat = geo.flatten(assembly_solids)

    # plt.figure()
    # geo.draw(assembly_solids)

    radians = np.linspace(0, np.pi, 100)
    arc_radians = np.linspace(-np.pi / 8, np.pi / 8, 100)
    source, detector_points, extent = geo.fan_beam_paths(60, arc_radians, radians, extent=True)
    source = source[0, :, :]

    # grid = geo.Grid(width=25, height=15, num_x=50, num_y=30)
    grid = geo.Grid(width=25, height=15, num_x=25, num_y=15)
    # grid.draw()

    cell_i = 109
    grid_points = grid.cell(cell_i)

    # super sample
    supersample = 4
    min_gridx, max_gridx = grid.points[0, 0, 0], grid.points[0, -1, 0]
    min_gridy, max_gridy = grid.points[-1, 0, 1], grid.points[0, 0, 1]
    numx = supersample * (np.size(grid.points, 1) - 1) + 2
    numy = supersample * (np.size(grid.points, 0) - 1) + 2
    xs = np.linspace(min_gridx, max_gridx, numx)[1:-1]
    ys = np.linspace(min_gridy, max_gridy, numy)[1:-1]
    fissionimage = np.zeros((np.size(xs, 0), np.size(ys, 0)), dtype=np.double)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            fissionimage[i, j] = fission.find_fissionval_at_point(np.array([x, y]), assembly_flat)

    # down sample
    dims = (int(np.size(fissionimage, 0) / supersample), int(np.size(fissionimage, 1) / supersample))
    fissionimage_down = rebin(fissionimage, dims)

    plt.figure()
    plt.imshow(fissionimage.T, interpolation='none', extent=extent)
    # plt.colorbar()
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')

    plt.figure()
    plt.imshow(fissionimage_down.T, interpolation='none', extent=extent)
    # plt.colorbar()
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')

    # plt.fill(grid_points[:, 0], grid_points[:, 1], color='red', alpha=0.5, zorder=12)

    response_single = np.load(r'data\fission_response_single.npy')
    response_double = np.load(r'data\fission_response_double.npy')

    single_probs = np.load(r'data\single_probs.npy')
    double_probs = np.load(r'data\double_probs.npy')


    def project_fission(fission_image, response):
        response_shape = response.shape
        rr = response.reshape(-1, np.size(response, 1) * np.size(response, 2))
        image = fission_image.T.flatten()
        projection = np.dot(image, rr).reshape(response_shape[1], response_shape[2])
        return projection


    single_project = project_fission(fissionimage_down, response_single)

    plt.figure()
    plt.imshow(single_project.T, interpolation='none', extent=extent)
    # plt.colorbar()
    plt.title('Single Neutron Forward Projection')
    plt.xlabel('Detector Orientation')
    plt.ylabel('Relative Neutron Angle')
    plt.tight_layout()

    plt.figure()
    plt.imshow(single_probs.T, interpolation='none', extent=extent)
    # plt.colorbar()
    plt.title('Single Neutron Probability')
    plt.xlabel('Detector Orientation')
    plt.ylabel('Relative Neutron Angle')
    plt.tight_layout()

    double_project = project_fission(fissionimage_down, response_double)

    plt.figure()
    plt.imshow(double_project.T, interpolation='none', extent=extent)
    # plt.colorbar()
    plt.title('Double Neutron Forward Projection')
    plt.xlabel('Detector Orientation')
    plt.ylabel('Relative Neutron Angle')
    plt.tight_layout()

    plt.figure()
    plt.imshow(double_probs.T, interpolation='none', extent=extent)
    # plt.colorbar()
    plt.title('Double Neutron Probability')
    plt.xlabel('Detector Orientation')
    plt.ylabel('Relative Neutron Angle')
    plt.tight_layout()

    plt.show()
