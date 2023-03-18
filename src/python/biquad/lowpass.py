"""
Copyright (c) 2023 Juergen Hock

SPDX-License-Identifier: MIT

Source: https://github.com/jurihock/biquad
"""

from .biquad import biquad, __df1__, __gain__, __resize__

import numba
import numpy


class lowpass(biquad):
    """
    Lowpass filter (LPF).
    """

    def __init__(self, sr, f=None, g=0, q=0.7071):
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

        super().__init__(sr=sr, f=f, g=g, q=q)

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
        g = __resize__(self.g if g is None else __gain__(g), x.shape)
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

            alpha = sinw / (+2 * q[i])
            beta  = cosw * (-2)

            # update b
            ba[0, 1] = +(1 - cosw)
            ba[0, 2] = ba[0, 1] / +2
            ba[0, 0] = ba[0, 2]

            # update a
            ba[1, 0] = 1 + alpha
            ba[1, 1] =     beta
            ba[1, 2] = 1 - alpha

            # update y
            __df1__(g[i], ba, xy, x, y, i)
