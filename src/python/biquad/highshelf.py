from .biquad import biquad, __df1__

import numba
import numpy


class highshelf(biquad):
    """
    Highshelf filter (HSF).
    """

    def __init__(self, sr, gain=6, q=1):

        super().__init__(sr, q)

        self.gain = gain
        self.amp = 10 ** (gain / 40)

        self.__call__(0, 1) # warmup numba

    def __call__(self, x, f, q=None):
        """
        Process single or multiple samples at once.
        """

        scalar = numpy.isscalar(x)

        ba = self.ba
        xy = self.xy

        x = numpy.atleast_1d(x)
        y = numpy.zeros(x.shape, x.dtype)

        f = numpy.atleast_1d(f)
        q = numpy.atleast_1d(q or self.q)

        f = numpy.resize(f, x.shape)
        q = numpy.resize(q, x.shape)

        sr = self.sr
        amp = self.amp

        self.__filter__(ba, xy, x, y, f, q, sr, amp)

        return y[0] if scalar else y

    @staticmethod
    @numba.jit(nopython=True, fastmath=True)
    def __filter__(ba, xy, x, y, f, q, sr, amp):

        sq = 2 * numpy.sqrt(amp)
        rs = 2 * numpy.pi / sr

        for i in range(x.size):

            w = f[i] * rs

            cosw = numpy.cos(w)
            sinw = numpy.sin(w)

            p = sq * sinw / (2 * q[i])

            pamp = amp + 1
            namp = amp - 1

            # update b
            ba[0, 0] = (pamp + namp * cosw + p) * amp
            ba[0, 1] = (namp + pamp * cosw)     * amp * -2
            ba[0, 2] = (pamp + namp * cosw - p) * amp

            # update a
            ba[1, 0] = (pamp - namp * cosw + p)
            ba[1, 1] = (namp - pamp * cosw)           * +2
            ba[1, 2] = (pamp - namp * cosw - p)

            # update y
            __df1__(ba, xy, x, y, i)
