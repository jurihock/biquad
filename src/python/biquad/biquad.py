"""
Copyright (c) 2023 Juergen Hock

SPDX-License-Identifier: MIT

Source: https://github.com/jurihock/biquad
"""

import numba
import numpy


@numba.jit(nopython=True, fastmath=True)
def __df1__(g, ba, xy, x, y, i):
    """
    Compute filter output y[i] based on filter input x[i]
    as well as specified filter gain g, coeffs ba and delay line xy,
    according to the Direct Form 1.
    """

    # roll x
    xy[0, 2] = xy[0, 1]
    xy[0, 1] = xy[0, 0]

    # roll y
    xy[1, 2] = xy[1, 1]
    xy[1, 1] = xy[1, 0]

    # update x
    xy[0, 0] = x[i]

    # compute intermediate results b*x and a*y
    bx = ba[0, 0] * xy[0, 0] + ba[0, 1] * xy[0, 1] + ba[0, 2] * xy[0, 2]
    ay =                       ba[1, 1] * xy[1, 1] + ba[1, 2] * xy[1, 2]

    # update y
    xy[1, 0] = (bx * g - ay) / ba[1, 0]

    # return y
    y[i] = xy[1, 0]


def __gain__(x, divisor=20):
    """
    Convert the specified decibel gain to the linear gain.
    """

    if numpy.isscalar(x):

        y = x / divisor
        y = 10 ** y

    else:

        y = numpy.atleast_1d(x) / divisor
        numpy.power(10, y, out=y)

    return y


def __resize__(x, shape):
    """
    Resize the specified value to the ndarray of desired shape.
    """

    if numpy.isscalar(x):

        y = numpy.full(shape, x)

    else:

        x = numpy.atleast_1d(x)
        y = numpy.resize(x, shape)

    return y


class biquad:
    """
    Biquad filter base class.
    """

    ba = numpy.array([[1, 0, 0], [1, 0, 0]], float)
    """
    Biquad filter coefficient matrix of shape (2, 3) excl. gain factor:
        - ba[0] holds b coefficients
        - ba[1] holds a coefficients
    """

    xy = numpy.array([[0, 0, 0], [0, 0, 0]], float)
    """
    Biquad filter delay line matrix of shape (2, 3):
        - xy[0] holds input values
        - xy[1] holds output values
    """

    def __init__(self, sr, f=None, g=None, q=None):
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

        assert (sr is not None) and (numpy.isscalar(sr) and numpy.isreal(sr))
        assert (f  is     None) or  (numpy.isscalar(f)  and numpy.isreal(f))
        assert (g  is     None) or  (numpy.isscalar(g)  and numpy.isreal(g))
        assert (q  is     None) or  (numpy.isscalar(q)  and numpy.isreal(q))

        self.sr = sr
        self.f  = (sr / 4) if f is None else f
        self.g  = __gain__(0 if g is None else g)
        self.q  = 1 if q is None else q

        # warmup numba
        g  = self.g
        ba = self.ba
        xy = self.xy
        x  = numpy.zeros(1, float)
        y  = numpy.zeros(x.shape, x.dtype)
        __df1__(g, ba, xy, x, y, 0)

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

        self.__filter__(ba, xy, x, y, g)

        self.f = f[-1]
        self.g = g[-1]
        self.q = q[-1]

        return y[0] if scalar else y

    @staticmethod
    @numba.jit(nopython=True, fastmath=True)
    def __filter__(ba, xy, x, y, g):

        for i in range(x.size):

            __df1__(g[i], ba, xy, x, y, i)

    def response(self, *, norm=False, log=False):
        """
        Returns frequency and phase response of the transfer function given by the ba coefficients.

        Parameters
        ----------
        norm : bool, optional
            Option whether to normalize the output frequency response.
        log : bool, optional
            Option whether to express the output frequency values logarithmically.

        Returns
        -------
        w : array
            Corresponding frequency values.
        h : array
            Complex filter response values.

        See also
        --------
            scipy.signal.freqz
        """

        (b, a), sr = self.ba, self.sr

        b *= self.g

        n = int(sr / 2)

        # compute frequencies from 0 to pi or sr/2 but excluding the Nyquist frequency
        w = numpy.linspace(0, numpy.pi, n, endpoint=False) \
            if not log else \
            numpy.logspace(numpy.log10(1), numpy.log10(numpy.pi), n, endpoint=False, base=10)

        # compute the z-domain transfer function
        z = numpy.exp(-1j * w)
        x = numpy.polynomial.polynomial.polyval(z, a, tensor=False)
        y = numpy.polynomial.polynomial.polyval(z, b, tensor=False)
        h = y / x

        # normalize frequency amplitudes
        h /= len(h) if norm else 1

        # normalize frequency values according to sr
        w = (w * sr) / (2 * numpy.pi)

        return w, h

    def plot(self, *, log=False):
        """
        Creates filter frequency and phase response plot.

        Parameters
        ----------
        log : bool, optional
            Option whether to express frequency values logarithmically.
        """

        import matplotlib.pyplot as pyplot

        w, h = self.response()

        with numpy.errstate(divide='ignore', invalid='ignore'):
            habs = 20 * numpy.log10(numpy.abs(h))

        harg = numpy.angle(h)

        pyplot.plot(w, habs, color='b', alpha=0.9)
        axis1 = pyplot.gca()
        axis2 = axis1.twinx()
        axis2.plot(w, harg, color='g', alpha=0.9)

        axis1.set_xlabel('Hz')
        axis1.set_ylabel('dB',  color='b')
        axis2.set_ylabel('rad', color='g')

        habsmin = numpy.min(habs[numpy.isfinite(habs)])
        habsmax = numpy.max(habs[numpy.isfinite(habs)])

        habsmin = numpy.minimum(habsmin, -100)
        habsmax = numpy.maximum(habsmax, +10)

        hargmin = -numpy.pi
        hargmax = +numpy.pi

        axis1.set_ylim((habsmin * 1.1, habsmax * 1.1))
        axis2.set_ylim((hargmin * 1.1, hargmax * 1.1))

        if log: axis1.set_xscale('log')

        return pyplot
