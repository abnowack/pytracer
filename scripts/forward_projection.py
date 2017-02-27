import sys
import matplotlib.pyplot as plt
import numpy as np
from scripts.assemblies import shielded_assembly
import pytracer.geometry as geo
import pytracer.transmission as transmission
import pytracer.algorithms as algorithms
import pytracer.fission as fission


def nice_double_plot(data1, data2, extent, title1='', title2='', xlabel='', ylabel=''):
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    # Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

    print(data1.min(), data1.max())
    im1 = ax1.imshow(data1, interpolation='none', extent=extent, cmap='viridis')
    ax1.set_title(title1)
    # vmin1, vmax1 = im1.get_clim()
    # print(vmin1, vmax1)

    im2 = ax2.imshow(data2, interpolation='none', extent=extent, cmap='viridis')
    ax2.set_title(title2)
    # vmin2, vmax2 = im2.get_clim()
    # im2.set_clim(vmin1, vmax1)
    # print(vmin2, vmax2)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.yaxis.labelpad = 40
    ax.set_frame_on(False)
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    plt.subplots_adjust(right=0.98, top=0.95, bottom=0.07, left=0.12, hspace=0.05, wspace=0.20)


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

    grid = geo.Grid(width=25, height=15, num_x=50, num_y=30)
    # grid = geo.Grid(width=25, height=15, num_x=25, num_y=15)
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
            fissionimage[i, j] = fission.fissionval_at_point(np.array([x, y]), assembly_flat)

    # down sample
    dims = (int(np.size(fissionimage, 0) / supersample), int(np.size(fissionimage, 1) / supersample))
    fissionimage_down = rebin(fissionimage, dims)

    plt.figure(figsize=(6, 4))
    plt.imshow(fissionimage.T, interpolation='none', extent=[-25. / 2, 25. / 2, -15. / 2, 15. / 2], cmap='viridis')
    plt.tight_layout()
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')

    plt.figure(figsize=(6, 4))
    plt.imshow(fissionimage_down.T, interpolation='none', extent=[-25. / 2, 25. / 2, -15. / 2, 15. / 2], cmap='viridis')
    plt.tight_layout()
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

    nice_double_plot(single_probs.T, single_project.T, extent, 'Single Measurement',
                     'Single Forward Project', 'Detector Orientation Angle', 'Source Neutron Direction Angle')

    double_project = project_fission(fissionimage_down, response_double)

    nice_double_plot(double_probs.T, double_project.T, extent, 'Double Measurement',
                     'Double Forward Project', 'Detector Orientation Angle', 'Source Neutron Direction Angle')

    plt.show()
