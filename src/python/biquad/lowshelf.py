"""
Copyright (c) 2023 Juergen Hock

SPDX-License-Identifier: MIT

Source: https://github.com/jurihock/biquad
"""

from .biquad import biquad, __df1__

import numba
import numpy


class lowshelf(biquad):
    """
    Lowshelf filter (LSF).
    """

    def __init__(self, sr, gain=6, *, f=None, q=1):
        """
        Create a new filter instance.

        Parameters
        ----------
        sr : int or float
            Sample rate in hertz.
        gain : int or float, optional
            Filter peak gain value in decibel.
        f : int or float, optional
            Persistent filter frequency parameter in hertz.
        q : int or float, optional
            Persistent filter quality parameter.
        """

        super().__init__(sr=sr, f=f, q=q)

        self.gain = gain
        self.amp = 10 ** (gain / 40)

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
            ba[0, 0] = (pamp - namp * cosw + p) * amp
            ba[0, 1] = (namp - pamp * cosw)     * amp * +2
            ba[0, 2] = (pamp - namp * cosw - p) * amp

            # update a
            ba[1, 0] = (pamp + namp * cosw + p)
            ba[1, 1] = (namp + pamp * cosw)           * -2
            ba[1, 2] = (pamp + namp * cosw - p)

            # update y
            __df1__(ba, xy, x, y, i)
