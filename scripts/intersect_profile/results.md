Using just plain Python methods
```
   1471438 function calls in 6.747 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
   696640    5.085    0.000    5.085    0.000 math2d.py:12(intersect)
    12440    0.451    0.000    5.584    0.000 math2d.py:35(intersections)
    10000    0.441    0.000    6.715    0.001 math2d.py:48(attenuation_length)
    60556    0.197    0.000    0.376    0.000 linalg.py:1976(norm)
    50556    0.158    0.000    0.234    0.000 math2d.py:5(norm)
   166560    0.122    0.000    0.122    0.000 {numpy.core.multiarray.array}
   103553    0.109    0.000    0.109    0.000 {numpy.core.multiarray.dot}
     7560    0.037    0.000    0.037    0.000 {method 'reduce' of 'numpy.ufunc' objects}
    60556    0.031    0.000    0.039    0.000 numeric.py:406(asarray)
        1    0.029    0.029    6.747    6.747 radon_profiler.py:39(main)
```

Changing to Cython, but no additional changes
```
   1218898 function calls in 5.728 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
   696640    4.314    0.000    4.314    0.000 math2d_c.pyx:12(intersect)
    10000    0.504    0.000    5.701    0.001 math2d_c.pyx:48(attenuation_length)
    12440    0.327    0.000    4.641    0.000 math2d_c.pyx:35(intersections)
    50556    0.195    0.000    0.195    0.000 math2d_c.pyx:5(norm)
    60556    0.180    0.000    0.349    0.000 linalg.py:1976(norm)
    52997    0.038    0.000    0.038    0.000 {numpy.core.multiarray.dot}
     7560    0.036    0.000    0.036    0.000 {method 'reduce' of 'numpy.ufunc' objects}
    60556    0.029    0.000    0.036    0.000 numeric.py:406(asarray)
    52996    0.024    0.000    0.024    0.000 {method 'ravel' of 'numpy.ndarray' objects}
        1    0.020    0.020    5.728    5.728 radon_profiler.py:39(main)
```

Changing to Cython with working intersections conversion and low Python C-API overhead
```
   3036874 function calls (3029362 primitive calls) in 2.511 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    10000    1.159    0.000    2.391    0.000 math2d_c.pyx:74(attenuation_length)
   305808    0.217    0.000    0.321    0.000 stringsource:985(memoryview_fromslice)
    50208    0.183    0.000    0.371    0.000 linalg.py:1976(norm)
   390784    0.156    0.000    0.156    0.000 stringsource:341(__cinit__)
    50208    0.148    0.000    0.148    0.000 math2d_c.pyx:16(perp_norm)
   390784    0.083    0.000    0.083    0.000 stringsource:368(__dealloc__)
    12488    0.074    0.000    0.144    0.000 math2d_c.pyx:26(intersections)
     7512    0.044    0.000    0.044    0.000 {method 'reduce' of 'numpy.ufunc' objects}
```

After converting attenuation_length efficiently to Cython as well
```
   1215050 function calls in 0.552 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    10000    0.158    0.000    0.451    0.000 math2d_c.pyx:69(attenuation_length)
   154928    0.069    0.000    0.069    0.000 stringsource:341(__cinit__)
    12488    0.058    0.000    0.121    0.000 math2d_c.pyx:18(intersections)
    69952    0.042    0.000    0.061    0.000 stringsource:985(memoryview_fromslice)
    60000    0.031    0.000    0.085    0.000 stringsource:395(__getitem__)
    84976    0.025    0.000    0.080    0.000 stringsource:643(memoryview_cwrapper)
   154928    0.025    0.000    0.025    0.000 stringsource:368(__dealloc__)
    10000    0.024    0.000    0.531    0.000 {math2d_c.attenuation_length}
    60000    0.023    0.000    0.023    0.000 stringsource:896(pybuffer_index)
        1    0.019    0.019    0.552    0.552 radon_profiler.py:63(main)
```

After caching intersects and indexes so we dont need to keep creating new arrays everytime
```
   495292 function calls in 0.201 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    10000    0.033    0.000    0.077    0.000 math2d_c.pyx:73(attenuation_length)
    84976    0.031    0.000    0.031    0.000 stringsource:341(__cinit__)
    10000    0.027    0.000    0.167    0.000 {math2d_c.attenuation_length}
        1    0.018    0.018    0.201    0.201 radon_profiler.py:68(main)
    60000    0.016    0.000    0.043    0.000 stringsource:643(memoryview_cwrapper)
    12488    0.015    0.000    0.034    0.000 math2d_c.pyx:19(intersections)
    24976    0.014    0.000    0.020    0.000 stringsource:985(memoryview_fromslice)
    10012    0.014    0.000    0.014    0.000 {numpy.core.multiarray.array}
    10000    0.012    0.000    0.093    0.000 math2d_c.pyx:73(attenuation_length (wrapper))
```

After having intersections just return the number of intersections stored in the cache
```
         345436 function calls in 0.140 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    60000    0.025    0.000    0.025    0.000 stringsource:341(__cinit__)
    10000    0.025    0.000    0.110    0.000 {math2d_c.attenuation_length}
    10000    0.016    0.000    0.027    0.000 math2d_c.pyx:73(attenuation_length)
        1    0.016    0.016    0.140    0.140 radon_profiler.py:68(main)
    60000    0.015    0.000    0.040    0.000 stringsource:643(memoryview_cwrapper)
    10012    0.012    0.000    0.012    0.000 {numpy.core.multiarray.array}
    10000    0.012    0.000    0.042    0.000 math2d_c.pyx:73(attenuation_length (wrapper))
    12488    0.009    0.000    0.009    0.000 math2d_c.pyx:19(intersections)
```

Turning profiling off then reduces this
```
         20252 function calls in 0.092 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    10000    0.063    0.000    0.063    0.000 {math2d_c.attenuation_length}
        1    0.015    0.015    0.092    0.092 radon_profiler.py:68(main)
    10012    0.012    0.000    0.012    0.000 {numpy.core.multiarray.array}
```

Instead of calling attenuation calculation multiple times from Python, create a function in Cython which iterates
over a list of paths and save attenuation in pre-allocated array, doing all attenuation paths at once
```
   253 function calls in 0.019 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.016    0.016    0.016    0.016 {math2d_c.calc_attenuation_bulk}
        1    0.002    0.002    0.002    0.002 radon_profiler.py:26(radon_scan_points)
        1    0.001    0.001    0.001    0.001 {numpy.core.multiarray.dot}
       19    0.000    0.000    0.000    0.000 {numpy.core.multiarray.zeros}
```

Ridiculously fast! We went from 6.747 to 0.019 seconds, a 350x speed up!