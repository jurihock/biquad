import os, sys
src = os.path.join(os.path.dirname(__file__), '..', 'src', 'python')
sys.path.insert(0, src)

import biquad
import matplotlib.pyplot as plot


sr = 44100

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

    fltr = biquad.filter(name, sr)

    fltr(0, sr / 4)

    figure = plot.figure()
    figure.suptitle(name)

    plot.subplot(1, 2, 1)
    fltr.plot().frequency()

    plot.subplot(1, 2, 2)
    fltr.plot().phase()

    plot.tight_layout()
    plot.show()
