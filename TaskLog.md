10/3/16

Speed up fission calculation routines used by fission.scan as used in tests/fission_measurement.py

Before starting work

"C:\Users\Aaron Nowack\Anaconda3\python.exe" H:/GitHub/pytracer/scripts/fission_measurement.py
         3653215 function calls in 15.223 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    57420    5.754    0.000    5.754    0.000 {built-in method pytracer.transmission_c.absorbances}
    57420    1.834    0.000    3.826    0.000 geometry.py:24(solid_angle)
   246000    1.102    0.000    1.102    0.000 {method 'reduce' of 'numpy.ufunc' objects}
   172260    0.883    0.000    1.993    0.000 linalg.py:1976(norm)
    20000    0.502    0.000    1.109    0.000 fission.py:46(find_fission_segments)
    57420    0.488    0.000    0.818    0.000 shape_base.py:785(tile)
    57420    0.487    0.000    0.487    0.000 {built-in method pytracer.transmission_c.absorbance}
    57420    0.485    0.000    0.485    0.000 geometry.py:6(center)
    57420    0.440    0.000   12.191    0.000 fission.py:113(probability_detect)
    11484    0.386    0.000   13.972    0.001 fission.py:155(probability_segment_neutron)
    57420    0.324    0.000   12.515    0.000 fission.py:136(probability_out)

Optimization Targets:
probability_path_neutron ---> find_fission_segments
                          |-> probability_segment_neutron -> probability_segment_neutron
                              -> probability_out -> probability_detect -> geometry/solid_angle

After cythonizing geometry/solid_angle

"C:\Users\Aaron Nowack\Anaconda3\python.exe" H:/GitHub/pytracer/scripts/fission_measurement.py
         2389975 function calls in 12.206 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    57420    5.751    0.000    5.751    0.000 {built-in method pytracer.transmission_c.absorbances}
    57420    0.637    0.000    0.637    0.000 {built-in method pytracer.geometry_c.solid_angle}
    57420    0.523    0.000    0.523    0.000 geometry.py:9(center)
    20000    0.510    0.000    1.146    0.000 fission.py:46(find_fission_segments)
    57420    0.470    0.000    0.470    0.000 {built-in method pytracer.transmission_c.absorbance}
    57420    0.470    0.000    0.785    0.000 shape_base.py:785(tile)
    57420    0.457    0.000    9.212    0.000 fission.py:113(probability_detect)
    11484    0.366    0.000   10.919    0.001 fission.py:155(probability_segment_neutron)
    73740    0.322    0.000    0.322    0.000 {method 'reduce' of 'numpy.ufunc' objects}
    57420    0.316    0.000    9.528    0.000 fission.py:136(probability_out)
    57420    0.253    0.000    6.307    0.000 transmission.py:64(attenuations)
   154840    0.244    0.000    0.244    0.000 {built-in method numpy.core.multiarray.array}
    16320    0.233    0.000    0.233    0.000 fission.py:52(point_is_outer_segment_side)
    57420    0.180    0.000    0.737    0.000 transmission.py:44(attenuation)