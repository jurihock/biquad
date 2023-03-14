from .plot import plot

import numba
import numpy


@numba.jit(nopython=True, fastmath=True)
def __df1__(ba, xy, x, y, i):
    """
    Computes filter output y[i] based on filter input x[i]
    as well as specified filter coeffs ba and delay line xy,
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

    # update y
    xy[1, 0] = (ba[0, 0] * xy[0, 0]  + \
                ba[0, 1] * xy[0, 1]  + \
                ba[0, 2] * xy[0, 2]  - \
                ba[1, 1] * xy[1, 1]  - \
                ba[1, 2] * xy[1, 2]) / ba[1, 0]

    # return y
    y[i] = xy[1, 0]


class biquad:
    """
    Biquad filter base class.
    """

    ba = numpy.array([[1, 0, 0], [1, 0, 0]], float)
    """
    Biquad filter coefficient matrix of shape (2, 3):
        - ba[0] holds b coefficients
        - ba[1] holds a coefficients
    """

    xy = numpy.array([[0, 0, 0], [0, 0, 0]], float)
    """
    Biquad filter delay line matrix of shape (2, 3):
        - xy[0] holds input values
        - xy[0] holds output values
    """

    def __init__(self, sr, q=None):
        """
        Create a new filter instance.

        Parameters
        ----------
        sr : int or float
            Sample rate in hertz.
        q : int or float, optional
            Persistent filter quality parameter.
        """

        assert (sr is not None) and (numpy.isscalar(sr) and numpy.isreal(sr))
        assert (q  is     None) or  (numpy.isscalar(q)  and numpy.isreal(q))

        self.sr = sr
        self.q  = q

        # warmup numba
        ba = self.ba
        xy = self.xy
        x = numpy.zeros(1, float)
        y = numpy.zeros(x.shape, x.dtype)
        __df1__(ba, xy, x, y, 0)

    def __call__(self, x, *args, **kwargs):
        """
        Process single or multiple contiguous signal values at once.

        Parameters
        ----------
        x : scalar or array like
            Filter input data.

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

        self.__filter__(ba, xy, x, y)

        return y[0] if scalar else y

    @staticmethod
    @numba.jit(nopython=True, fastmath=True)
    def __filter__(ba, xy, x, y):

        for i in range(x.size):

            __df1__(ba, xy, x, y, i)

    def response(self, norm=False, log=False):
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
        """
        Returns a filter response plotting wrapper to
        easily create the frequency or phase response plots.
        """

        w, h = self.response()

        return plot(w, h)
