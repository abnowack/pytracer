import sys
import matplotlib.pyplot as plt
import numpy as np
from scripts.assemblies import shielded_assembly
import pytracer.geometry as geo
import pytracer.transmission as transmission
import pytracer.algorithms as algorithms
import pytracer.fission as fission

if __name__ == "__main__":
    # assembly_solids = shielded_assembly()
    # assembly_flat = geo.flatten(assembly_solids)

    response_single = np.load(r'data\fission_response_single.npy')
    response_double = np.load(r'data\fission_response_double.npy')
    single_probs = np.load(r'data\single_probs.npy')
    double_probs = np.load(r'data\double_probs.npy')

    rr1 = response_single.reshape(-1, np.size(response_single, 2))
    mm1 = single_probs.reshape((-1))
    inv1 = np.dot(rr1.T, rr1)
    inv1 = np.linalg.inv(inv1)
    rhs1 = np.dot(rr1.T, mm1.T)
    ans1 = np.dot(inv1, rhs1)

    rr2 = response_double.reshape(-1, np.size(response_double, 2))
    mm2 = double_probs.reshape((-1))
    inv2 = np.dot(rr2.T, rr2)
    inv2 = np.linalg.inv(inv2)
    rhs2 = np.dot(rr2.T, mm2.T)
    ans2 = np.dot(inv2, rhs2)

    print('hello')
