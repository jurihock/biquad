"""
Copyright (c) 2023 Juergen Hock

SPDX-License-Identifier: MIT

Source: https://github.com/jurihock/biquad
"""

from .biquad import biquad, __df1__

import numba
import numpy


class allpass(biquad):
    """
    Allpass filter (APF).
    """

    def __init__(self, sr, *, f=None, q=1):
        """
        Create a new filter instance.

        Parameters
        ----------
        sr : int or float
            Sample rate in hertz.
        f : int or float, optional
            Persistent filter frequency parameter in hertz.
        q : int or float, optional
            Persistent filter quality parameter.
        """

        super().__init__(sr=sr, f=f, q=q)

        self.__call__(0, 1) # warmup numba

    def __call__(self, x, f=None, q=None):
        """
        Process single or multiple contiguous signal values at once.

        Parameters
        ----------
        x : scalar or array like
            Filter input data.
        f : scalar or array like, optional
            Instantaneous filter frequency parameter in hertz.
        q : scalar or array like, optional
            Instantaneous filter quality parameter.

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

        f = numpy.atleast_1d(self.f if f is None else f)
        q = numpy.atleast_1d(self.q if q is None else q)

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
            ba[0, 0] = 1 - p
            ba[0, 1] =     c
            ba[0, 2] = 1 + p

            # update a
            ba[1, 0] = 1 + p
            ba[1, 1] =     c
            ba[1, 2] = 1 - p

            # update y
            __df1__(ba, xy, x, y, i)
