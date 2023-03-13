import numpy


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

    def __init__(self, w, h):

        self.w = w
        self.h = h

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

        x, y = self.w, __abs__(self.h, db=True)

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

        x, y = self.w, __arg__(self.h, wrap=True)

        pyplot.plot(x, y)
        pyplot.xlabel('Hz')
        pyplot.ylabel('rad')

        lim()

        return pyplot
