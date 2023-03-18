import os, sys
src = os.path.join(os.path.dirname(__file__), '..', 'src', 'python')
sys.path.insert(0, src)

import benchmark
import biquad
import numpy as np


def test():

    sr = 44100
    n  = 60*sr

    f = biquad.biquad(sr)
    x = np.arange(n).astype(float)

    with benchmark.measure():
        y = f(x)

    assert np.allclose(x, y)


if __name__ == '__main__':

    test()
