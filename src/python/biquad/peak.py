from .biquad import biquad

import numba
import numpy


class peak(biquad):
    """
    Peaking EQ filter.
    """

    def __init__(self, sr, gain=12, q=1):

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
        y = numpy.zeros(x.shape)

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

        rs = 2 * numpy.pi / sr

        for i in range(x.size):

            w = f[i] * rs

            cosw = numpy.cos(w)
            sinw = numpy.sin(w)

            p = sinw / (2 * q[i])

            m = p * amp
            d = p / amp

            # update b
            ba[0, 0] =  1 + m
            ba[0, 1] = -2 * cosw
            ba[0, 2] =  1 - m

            # update a
            ba[1, 0] = 1 + d
            ba[1, 1] = ba[0, 1]
            ba[1, 2] = 1 - d

            # roll x
            xy[0, 2] = xy[0, 1]
            xy[0, 1] = xy[0, 0]

            # roll y
            xy[1, 2] = xy[1, 1]
            xy[1, 1] = xy[1, 0]

            # update x and y
            xy[0, 0] = x[i]
            xy[1, 0] = (ba[0, 0] * xy[0, 0]  + \
                        ba[0, 1] * xy[0, 1]  + \
                        ba[0, 2] * xy[0, 2]  - \
                        ba[1, 1] * xy[1, 1]  - \
                        ba[1, 2] * xy[1, 2]) / ba[1, 0]

            y[i] = xy[1, 0]