"""
Copyright (c) 2023 Juergen Hock

SPDX-License-Identifier: MIT

Source: https://github.com/jurihock/biquad
"""

from .biquad import biquad, __df1__, __gain__, __resize__

import numba
import numpy


class highshelf(biquad):
    """
    Highshelf filter (HSF).
    """

    def __init__(self, sr, f=None, g=6, q=1):
        """
        Create a new filter instance.

        Parameters
        ----------
        sr : int or float
            Sample rate in hertz.
        f : int or float, optional
            Persistent filter frequency parameter in hertz.
        g : int or float, optional
            Persistent filter gain parameter in decibel.
        q : int or float, optional
            Persistent filter quality parameter.
        """

        super().__init__(sr=sr, f=f, g=0, q=q)

        self.g = __gain__(g, 40)

        self.__call__(0) # warmup numba

    def __call__(self, x, f=None, g=None, q=None):
        """
        Process single or multiple contiguous signal values at once.

        Parameters
        ----------
        x : scalar or array like
            Filter input data.
        f : scalar or array like, optional
            Instantaneous filter frequency parameter in hertz.
        g : scalar or array like, optional
            Instantaneous filter gain parameter in decibel.
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

        f = __resize__(self.f if f is None else f, x.shape)
        g = __resize__(self.g if g is None else __gain__(g, 40), x.shape)
        q = __resize__(self.q if q is None else q, x.shape)

        sr = self.sr

        self.__filter__(ba, xy, x, y, f, g, q, sr)

        self.f = f[-1]
        self.g = g[-1]
        self.q = q[-1]

        return y[0] if scalar else y

    @staticmethod
    @numba.jit(nopython=True, fastmath=True)
    def __filter__(ba, xy, x, y, f, g, q, sr):

        rs = 2 * numpy.pi / sr

        for i in range(x.size):

            w = f[i] * rs

            cosw = numpy.cos(w)
            sinw = numpy.sin(w)

            alpha = sinw / (2 * q[i])
            alpha *= 2 * numpy.sqrt(g[i])

            plus  = g[i] + 1
            minus = g[i] - 1

            # update b
            ba[0, 0] = (plus  + minus * cosw + alpha)
            ba[0, 1] = (minus + plus  * cosw) * -2
            ba[0, 2] = (plus  + minus * cosw - alpha)

            # update a
            ba[1, 0] = (plus  - minus * cosw + alpha)
            ba[1, 1] = (minus - plus  * cosw) * +2
            ba[1, 2] = (plus  - minus * cosw - alpha)

            # update y
            __df1__(g[i], ba, xy, x, y, i)
