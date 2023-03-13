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

    def __init__(self, sr):

        self.sr = sr

    def plot(self):

        b, a = self.ba
        sr   = self.sr

        return plot(b, a, sr)

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
