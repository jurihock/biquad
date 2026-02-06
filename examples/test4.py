import os, sys
src = os.path.join(os.path.dirname(__file__), '..', 'src', 'python')
sys.path.insert(0, src)

import biquad
import numpy as np

lp =  biquad.lowpass(48000, f=5000, q=1)
hp =  biquad.highpass(48000, f=5000, q=1)

assert not np.allclose(lp.ba, hp.ba) #2
