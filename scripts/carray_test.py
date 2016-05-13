import numpy as np
import sys

from intersect_module import intersecting_segments_c as isc

if __name__ == "__main__":
    a = np.ones((3, 4, 5))
    print a.shape, sys.getrefcount(a)
    print [sys.getrefcount(i) for i in a.shape]
    print isc(a), sys.getrefcount(a)
    print [sys.getrefcount(i) for i in a.shape]
    print a.shape, sys.getrefcount(a)
    print [sys.getrefcount(i) for i in a.shape]

    a.shape