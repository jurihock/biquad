import os, sys
src = os.path.join(os.path.dirname(__file__), '..', 'src', 'python')
sys.path.insert(0, src)

import biquad
import matplotlib.pyplot as plot


def test(name):

    sr = 44100

    f = biquad.filter(name, sr, f=sr/4)

    figure = plot.figure()
    figure.suptitle(name)

    plot.subplot(1, 2, 1)
    f.plot().frequency()

    plot.subplot(1, 2, 2)
    f.plot().phase()

    plot.tight_layout()
    plot.show()


if __name__ == '__main__':

    filters = [
        'biquad',
        'apf',
        'bpf',
        'hpf',
        'hsf',
        'lpf',
        'lsf',
        'notch',
        'peak',
    ]

    for name in filters:

        test(name)
