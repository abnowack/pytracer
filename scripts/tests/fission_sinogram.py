"""
Create response matrices for the single and double fission responses over a grid, for the basic assembly geometry.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scripts.assemblies import shielded_assembly
import pytracer.geometry as geo
import pytracer.fission as fission


def nice_double_plot(data1, data2, extent, title1='', title2='', xlabel='', ylabel='', cbar=False, cval_min=0,
                     cval_max=1):
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
    if cbar:
        im1 = ax1.imshow(data1, interpolation='none', extent=extent, cmap='viridis', vmin=cval_min, vmax=cval_max)
        fig.colorbar(im1, ax=ax1)
    else:
        im1 = ax1.imshow(data1, interpolation='none', extent=extent, cmap='viridis')

    ax1.set_title(title1)

    if cbar:
        im2 = ax2.imshow(data2, interpolation='none', extent=extent, cmap='viridis', vmin=cval_min, vmax=cval_max)
        fig.colorbar(im2, ax=ax2)
    else:
        im2 = ax2.imshow(data2, interpolation='none', extent=extent, cmap='viridis')

    ax2.set_title(title2)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.yaxis.labelpad = 40
    ax.set_frame_on(False)
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    plt.subplots_adjust(right=0.98, top=0.95, bottom=0.07, left=0.12, hspace=0.05, wspace=0.20)


if __name__ == "__main__":
    assembly_solids = shielded_assembly()
    assembly_flat = geo.flatten(assembly_solids)

    # nu_dist1 = np.array([0.0481677, 0.2485215, 0.4253044, 0.2284094, 0.0423438, 0.0072533], dtype=np.double)
    # nu_dist2 = np.array([0.0481677, 0.2485215, 0.4253044, 0.2284094, 0.0423438, 0.0072533], dtype=np.double)
    nu_dist1 = np.array([0.00042589464915587262, 0.95377787638758305, 0.011600774352912844,
                         0.014628422095961826, 0.010559456902820319, 0.0045704616984181034,
                         0.0019929358022211545, 0.0010654489210268395, 0.00061290526855218642,
                         0.00032767568379860876, 0.0001829408658320835, 0.00010557751346789126,
                         6.1187214507709063e-05, 3.5691594904520266e-05, 2.1071019882219079e-05,
                         1.2551130623148222e-05, 7.5170528174737343e-06, 4.5288491906447061e-06,
                         2.7440005591414875e-06, 1.6703058124665261e-06, 1.0209159280578123e-06,
                         6.2639470008860228e-07, 3.856687388804275e-07, 2.3819649275273812e-07,
                         1.4753634408513646e-07, 9.1623837649893816e-08, 5.7038926842413833e-08,
                         3.5588246454745566e-08, 2.2250718463272138e-08, 1.3938638081478588e-08,
                         8.7473818343146544e-09], dtype=np.double)
    nu_dist2 = np.array([0.0016700659968829446, 0.81885915954299582, 0.033695878005496935,
                         0.03850180875978048, 0.028301897432790997, 0.016893099727063012,
                         0.011898867719846587, 0.0090002234193772672, 0.0068900712527471706,
                         0.0053868483031017586, 0.004318679540168539, 0.0035121223814463223,
                         0.0028908207168062729, 0.0024060832740782551, 0.002020972715646972,
                         0.0017107229223443496, 0.0014580282854617962, 0.0012501287392593882,
                         0.0010775546129419946, 0.00093319289041558003, 0.00081159651564652715,
                         0.00070854006624186704, 0.0006207104403290586, 0.00054548133223521179,
                         0.00048075062977389314, 0.00042482134116017084, 0.00037631265078377692,
                         0.00033409283952785525, 0.00029722818836840846, 0.00026494362663799173,
                         0.00023659214252475999], dtype=np.double)
    nu_dist1 /= np.sum(nu_dist1)
    nu_dist2 /= np.sum(nu_dist2)

    plt.figure()
    geo.draw(assembly_solids)

    radians = np.linspace(0, np.pi, 100)
    arc_radians = np.linspace(-np.pi / 8, np.pi / 8, 100)
    source, detector_points, extent = geo.fan_beam_paths(60, arc_radians, radians, extent=True)
    source = source[0, :, :]

    plt.xlabel('X (cm)', size=18)
    plt.ylabel('Y (cm)', size=18)
    ax = plt.gca()
    ax.set_aspect('equal')

    single_probs1 = fission.scan(source, detector_points, detector_points, assembly_flat, 1, nu_dist1)
    single_probs2 = fission.scan(source, detector_points, detector_points, assembly_flat, 1, nu_dist2)

    double_probs1 = fission.scan(source, detector_points, detector_points, assembly_flat, 2, nu_dist1)
    double_probs2 = fission.scan(source, detector_points, detector_points, assembly_flat, 2, nu_dist2)

    maxval = max([single_probs1.max(), single_probs2.max(), double_probs1.max(), double_probs2.max()])

    nice_double_plot(single_probs1.T, single_probs2.T, extent, 'Single Neutron Probability 1',
                     'Single Neutron Probability 2', 'Detector Orientation Angle', 'Source Neutron Direction Angle',
                     True, 0, maxval)

    nice_double_plot(double_probs1.T, double_probs2.T, extent, 'Double Neutron Probability 1',
                     'Double Neutron Probability 2', 'Detector Orientation Angle', 'Source Neutron Direction Angle',
                     True, 0, maxval)

    nice_double_plot(single_probs1.T, double_probs1.T, extent, 'Single Neutron Probability 1',
                     'Double Neutron Probability 1', 'Detector Orientation Angle', 'Source Neutron Direction Angle',
                     True, 0, maxval)

    nice_double_plot(single_probs2.T, double_probs2.T, extent, 'Single Neutron Probability 2',
                     'Double Neutron Probability 2', 'Detector Orientation Angle', 'Source Neutron Direction Angle',
                     True, 0, maxval)

    plt.show()
