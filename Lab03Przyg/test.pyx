from cython.parallel import prange
from libc.stdio cimport printf

def simple_function():
    cdef int i = 0
    cdef int n = 30
    cdef int sums = 0

    for i in range(n):
        sums = sums + i

    print('abc')
    return sums