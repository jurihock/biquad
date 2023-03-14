from .biquad import biquad, __df1__

import numba
import numpy


class lowpass(biquad):
    """
    Lowpass filter (LPF).
    """

    def __init__(self, sr, q=0.7071):
        """
        Create a new filter instance.

        Parameters
        ----------
        sr : int or float
            Sample rate in hertz.
        q : int or float, optional
            Persistent filter quality parameter.
        """

        super().__init__(sr, q)

        self.__call__(0, 1) # warmup numba

    def __call__(self, x, f, q=None):
        """
        Process single or multiple contiguous signal values at once.

        Parameters
        ----------
        x : scalar or array like
            Filter input data.
        f : scalar or array like
            Instantaneous filter frequency parameter in hertz.
        q : scalar or array like, optional
            Optional instantaneous filter quality parameter.

        Returns
        -------
        y : scalar or ndarray
            Filter output data of the same shape and dtype as the input x.
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

        self.__filter__(ba, xy, x, y, f, q, sr)

        return y[0] if scalar else y

    @staticmethod
    @numba.jit(nopython=True, fastmath=True)
    def __filter__(ba, xy, x, y, f, q, sr):

        rs = 2 * numpy.pi / sr

        for i in range(x.size):

            w = f[i] * rs

            cosw = numpy.cos(w)
            sinw = numpy.sin(w)

            c = -(2 * cosw)
            p = sinw / (2 * q[i])

            # update b
            ba[0, 1] = +(1 - cosw)
            ba[0, 2] = ba[0, 1] / +2
            ba[0, 0] = ba[0, 2]

            # update a
            ba[1, 0] = 1 + p
            ba[1, 1] =     c
            ba[1, 2] = 1 - p

            # update y
            __df1__(ba, xy, x, y, i)
