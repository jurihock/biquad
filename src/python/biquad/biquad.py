from .plot import plot

import numba
import numpy


class biquad:
    """
    Biquad filter base class.
    """

    # filter coeffs
    ba = numpy.array([[1, 0, 0], [1, 0, 0]], float)

    # delay line
    xy = numpy.array([[0, 0, 0], [0, 0, 0]], float)

    def __init__(self, sr, q=None):

        assert (sr is not None) and (numpy.isscalar(sr) and numpy.isreal(sr))
        assert (q  is     None) or  (numpy.isscalar(q)  and numpy.isreal(q))

        self.sr = sr
        self.q  = q

    def __call__(self, x, *args, **kwargs):
        """
        Process single or multiple samples at once.
        """

        scalar = numpy.isscalar(x)

        ba = self.ba
        xy = self.xy

        x = numpy.atleast_1d(x)
        y = numpy.zeros(x.shape)

        self.__filter__(ba, xy, x, y)

        return y[0] if scalar else y

    @staticmethod
    @numba.jit(nopython=True, fastmath=True)
    def __filter__(ba, xy, x, y):

        for i in range(x.size):

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

    def response(self, norm=False, log=False):

        (b, a), sr = self.ba, self.sr

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

    def plot(self):

        w, h = self.response()

        return plot(w, h)
