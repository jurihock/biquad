import os, sys
src = os.path.join(os.path.dirname(__file__), '..', 'src', 'python')
sys.path.insert(0, src)

import benchmark
import biquad
import matplotlib.pyplot as plot
import numpy as np
import scipy


def db(x):

    y = np.abs(x)

    with np.errstate(divide='ignore', invalid='ignore'):
        return 20 * np.log10(y)


def fft(x, sr):

    w = np.fft.rfftfreq(x.size, 1/sr)
    y = x * np.hanning(x.size)
    h = np.fft.rfft(y, norm='forward')

    return w, h


def noise(n, s=1):

    lo = -1 / s
    hi = +1 / s

    return scipy.stats.truncnorm(lo, hi, loc=0, scale=s).rvs(n)


def test(name):

    sr = 44100
    n  = 1*sr

    f = biquad.filter(name, sr, f=sr/4, g=12)

    x = noise(n)

    with benchmark.measure():
        y = f(x)

    wx, hx = fft(x, sr)
    wy, hy = fft(y, sr)
    wf, hf = f.response()

    hx = db(hx)
    hy = db(hy)
    hf = db(hf)

    px = np.percentile(hx[np.isfinite(hx)], 99.9)
    py = np.percentile(hy[np.isfinite(hy)], 99.9)

    figure = plot.figure()
    figure.suptitle(name)

    plot.plot(wy, hy, color='gray', alpha=0.8, label='Output dft magnitude')
    plot.plot(wf, hf, color='blue', alpha=0.8, label='Filter transfer function')
    plot.axhline(y=px, color='magenta', alpha=0.5, linestyle='--', label='Input magnitude peak')
    plot.axhline(y=py, color='magenta', alpha=0.5, linestyle='-',  label='Output magnitude peak')

    plot.xlabel('Hz')
    plot.ylabel('dB')
    plot.legend()

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
