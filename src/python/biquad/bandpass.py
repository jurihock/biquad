"""
Copyright (c) 2023 Juergen Hock

SPDX-License-Identifier: MIT

Source: https://github.com/jurihock/biquad
"""

from .biquad import biquad, __df1__

import numba
import numpy


class bandpass(biquad):
    """
    Bandpass filter (BPF).
    """

    def __init__(self, sr, gain='skirt', *, f=None, q=0.7071):
        """
        Create a new filter instance.

        Parameters
        ----------
        sr : int or float
            Sample rate in hertz.
        gain : str, skirt or peak, optional
            Choice between constant skirt gain or constant 0 dB peak gain.
        f : int or float, optional
            Persistent filter frequency parameter in hertz.
        q : int or float, optional
            Persistent filter quality parameter.
        """

        super().__init__(sr=sr, f=f, q=q)

        assert gain in ['skirt', 'peak']

        self.gain = gain

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
        gain = self.gain

        self.__filter__(ba, xy, x, y, f, q, sr, gain)

        return y[0] if scalar else y

    @staticmethod
    @numba.jit(nopython=True, fastmath=True)
    def __filter__(ba, xy, x, y, f, q, sr, gain):

        rs = 2 * numpy.pi / sr
        skirt = gain == 'skirt'

        for i in range(x.size):

            w = f[i] * rs

            cosw = numpy.cos(w)
            sinw = numpy.sin(w)

            c = -(2 * cosw)
            p = sinw / (2 * q[i])
            g = sinw / 2 if skirt else p

            # update b
            ba[0, 0] = +g
            ba[0, 1] =  0
            ba[0, 2] = -g

            # update a
            ba[1, 0] = 1 + p
            ba[1, 1] =     c
            ba[1, 2] = 1 - p

            # update y
            __df1__(ba, xy, x, y, i)
