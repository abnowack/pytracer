# PyTracer
Active Neutron Interrogation modelling program

Features :
- Fast raytracing backend
- CT transmission modelling
- Neutron chain statistics calculation
- Neutron single, double, etc detection probability modelling
- Various reconstruciton methods (backprojection, L2, L1)

Build :
```
cd pytacer
python setup.py build_ext --inplace
```

[See Examples](./scripts/notebooks)

TODO :
- [ ] Implement Forward projection using fission probability image estimate
- [ ] Calculate Noonan derivatives
- [ ] Build Derivative Matrix
- [ ] Get Updated fission probability image estimate
- [ ] Use Occam's Inversion method for fission probability image reconstruction

TODO (Longer Term) :
- [ ] Implement a BVH tree for segment intersection speed up
- [ ] Convert raytracing backend to CUDA (optional)