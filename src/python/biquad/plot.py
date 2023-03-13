import numpy


def __response__(b, a, sr, norm=False, log=False):

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


def __abs__(x, db=False):

    if db:
        with numpy.errstate(divide='ignore', invalid='ignore'):
            return 20 * numpy.log10(numpy.abs(x))
    else:
        return numpy.abs(x)


def __arg__(x, wrap=None):

    if wrap is None:
        return numpy.angle(x)
    elif wrap:
        return (numpy.angle(x) + numpy.pi) % (2 * numpy.pi) - numpy.pi
    else:
        return numpy.unwrap(numpy.angle(x))


class plot:

    def __init__(self, b, a, sr):

        self.b = b
        self.a = a
        self.sr = sr

    def frequency(self, xlim=None, ylim=None):

        import matplotlib.pyplot as pyplot

        def lim():

            if xlim is not None:
                if isinstance(xlim, (list, tuple)):
                    pyplot.xlim(xlim)
                else:
                    pyplot.xlim(0, xlim)

            if ylim is not None:
                if isinstance(ylim, (list, tuple)):
                    pyplot.ylim(ylim)
                else:
                    pyplot.ylim(ylim, 0)
            else:
                pyplot.ylim(-110, numpy.maximum(10, y.max()))

        b  = self.b
        a  = self.a
        sr = self.sr

        x, y = __response__(b, a, sr)
        y    = __abs__(y, db=True)

        pyplot.plot(x, y)
        pyplot.xlabel('Hz')
        pyplot.ylabel('dB')

        lim()

        return pyplot

    def phase(self, xlim=None, ylim=None):

        import matplotlib.pyplot as pyplot

        def lim():

            if xlim is not None:
                if isinstance(xlim, (list, tuple)):
                    pyplot.xlim(xlim)
                else:
                    pyplot.xlim(0, xlim)

            if ylim is not None:
                if isinstance(ylim, (list, tuple)):
                    pyplot.ylim(ylim)
                else:
                    pyplot.ylim(-ylim, +ylim)
            else:
                pyplot.ylim(-numpy.pi, +numpy.pi)

        b  = self.b
        a  = self.a
        sr = self.sr

        x, y = __response__(b, a, sr)
        y    = __arg__(y, wrap=True)

        pyplot.plot(x, y)
        pyplot.xlabel('Hz')
        pyplot.ylabel('rad')

        lim()

        return pyplot
