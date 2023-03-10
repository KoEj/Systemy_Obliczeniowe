import numpy as np
cimport numpy as np
from cython.parallel import prange

np.import_array()
DTYPE = np.int
ctypedef np.int_t DTYPE_t

def prange_convolve(np.ndarray f, np.ndarray g):
    if g.shape[0] % 2 != 1 or g.shape[1] % 2 != 1:
        raise ValueError("Only odd dimensions on filter supported")
    assert f.dtype == DTYPE and g.dtype == DTYPE

    cdef int vmax = f.shape[0]
    cdef int wmax = f.shape[1]
    cdef int smax = g.shape[0]
    cdef int tmax = g.shape[1]
    cdef int smid = smax // 2
    cdef int tmid = tmax // 2
    cdef int xmax = vmax + 2 * smid
    cdef int ymax = wmax + 2 * tmid
    cdef np.ndarray h = np.zeros([xmax, ymax], dtype=DTYPE)
    cdef int x, y, s, t, v, w
    cdef int s_from, s_to, t_from, t_to
    cdef np.int_t value
    cdef int[:, :] f_view = f
    cdef int[:, :] g_view = g
    cdef int[:, :] h_view = h

    for x in prange(xmax, nogil=True):
        for y in prange(ymax):
            s_from = max(smid - x, -smid)
            s_to = min((xmax - x) - smid, smid + 1)
            t_from = max(tmid - y, -tmid)
            t_to = min((ymax - y) - tmid, tmid + 1)
            value = 0
            for s in prange(s_from, s_to):
                for t in prange(t_from, t_to):
                    v = x - smid + s
                    w = y - tmid + t
                    value += g_view[smid - s, tmid - t] * f_view[v, w]
            h_view[x, y] = value

    return h[0:x-1, 0:y-1]