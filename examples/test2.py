import os, sys
src = os.path.join(os.path.dirname(__file__), '..', 'src', 'python')
sys.path.insert(0, src)

import biquad
import matplotlib.pyplot as plot


def test(name):

    sr = 44100

    args = dict(f=sr/4)

    f = biquad.filter(name, sr, **args)

    figure = plot.figure()
    figure.suptitle(name)

    f.plot()

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
