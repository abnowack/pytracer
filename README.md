# PyTracer
Raytracing program written in Python, applying CT reconstruction methods

TODO :
- [x] Fix attenuation giving incorrect sign
- [x] Convert fission.py to Cython
- [x] Fast scan methods for doing singles and doubles measurements easily
- [x] Create single fission response

- [ ] Turn find_absorbance/fission at point functions / scripts into Cython functions

- [ ] Forward projection for fission
    - [ ] Split up into several parts
- [ ] Compare Normalization for transmission and fission forward projections
- [ ] Normalized reconstruction methods
- [ ] Plot fission reconstruction error for different nu around nu_true
- [ ]

Longer Term TODO :
- [ ] Implement a BVH tree for segment intersection speed up
- [ ] Convert to gpu through CUDA (optional backend)
- [ ] L1-norm reconstruction methods