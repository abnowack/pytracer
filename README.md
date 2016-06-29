# PyTracer
Raytracing program written in Python, applying CT reconstruction methods

FIX :
- [ ] Put faster attenuation calculations into main code
- [ ] Put Fission Segment Calculation into Cython
- [ ] Calc single fission response using above methods

TODO :
- [x] Get transmission CT reconstruction working again
- [x] Nice display of transmission response
- [x] Create Single Fission Response per pixel, save
- [ ] Create Full Single Fission Response Measurement
- [ ] Apply brute force reconstruction methods

Longer Term TODO :
- [ ] Implement a BVH tree for segment intersection speed up
- [ ] Convert to gpu through CUDA (optional backend)
- [ ] Upgrade to Python 3 (latest)
- [ ] Think about refactoring algorithm functions to input attenuation[in, out], fission[in, out] args
      instead of in_attenuation, out_attenuation, in_fission, out_fission