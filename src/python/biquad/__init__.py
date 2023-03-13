__version__ = "0.1"


from .allpass  import allpass
from .bandpass import bandpass
from .biquad   import biquad
from .highpass import highpass
from .lowpass  import lowpass
from .notch    import notch


def filter(name, sr, **kwargs):

    name = str(name).lower()

    if name in ['allpass', 'all', 'apf']:
        return allpass(sr, **kwargs)

    if name in ['bandpass', 'band', 'bpf']:
        return bandpass(sr, **kwargs)

    if name in ['highpass', 'high', 'hpf']:
        return highpass(sr, **kwargs)

    if name in ['lowpass', 'low', 'lpf']:
        return lowpass(sr, **kwargs)

    if name in ['notch', 'nf']:
        return notch(sr, **kwargs)

    return biquad(sr, **kwargs)
