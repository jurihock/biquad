from .biquad import biquad

import numba
import numpy


class bandpass(biquad):
    """
    Bandpass filter (BPF).
    """

    def __init__(self, sr, gain='skirt'):

        assert gain in ['skirt', 'peak']

        self.sr = sr
        self.gain = gain

        self.__call__(0, 1, 2) # warmup numba

    def __call__(self, x, f, bw):
        """
        Process single or multiple samples at once.
        """

        scalar = numpy.isscalar(x)

        ba = self.ba
        xy = self.xy

        x = numpy.atleast_1d(x)
        y = numpy.zeros(x.shape)

        f = numpy.atleast_1d(f)
        bw = numpy.atleast_1d(bw)

        f = numpy.resize(f, x.shape)
        bw = numpy.resize(bw, x.shape)

        sr = self.sr
        gain = self.gain

        self.__filter__(ba, xy, x, y, f, bw, sr, gain)

        return y[0] if scalar else y

    @staticmethod
    @numba.jit(nopython=True, fastmath=True)
    def __filter__(ba, xy, x, y, f, bw, sr, gain):

        ln = numpy.log(2) / 2
        rs = 2 * numpy.pi / sr
        skirt = gain == 'skirt'

        for i in range(x.size):

            w = f[i] * rs

            cosw = numpy.cos(w)
            sinw = numpy.sin(w)

            p = sinw * numpy.sinh(ln * bw[i] * w / sinw)
            g = sinw / 2 if skirt else p

            # update b
            ba[0, 0] = +g
            ba[0, 1] =  0
            ba[0, 2] = -g

            # update a
            ba[1, 0] =  1 + p
            ba[1, 1] = -2 * cosw
            ba[1, 2] =  1 - p

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
