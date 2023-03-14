__version__ = "0.2"


from .allpass   import allpass
from .bandpass  import bandpass
from .biquad    import biquad
from .highpass  import highpass
from .highshelf import highshelf
from .lowpass   import lowpass
from .lowshelf  import lowshelf
from .notch     import notch
from .peak      import peak


def filter(name, sr, **kwargs):

    name = str(name).lower()

    if name in ['allpass', 'all', 'ap', 'apf']:
        return allpass(sr, **kwargs)

    if name in ['bandpass', 'band', 'bp', 'bpf']:
        return bandpass(sr, **kwargs)

    if name in ['highpass', 'high', 'hp', 'hpf']:
        return highpass(sr, **kwargs)

    if name in ['highshelf', 'hs', 'hsf']:
        return highshelf(sr, **kwargs)

    if name in ['lowpass', 'low', 'lp', 'lpf']:
        return lowpass(sr, **kwargs)

    if name in ['lowshelf', 'ls', 'lsf']:
        return lowshelf(sr, **kwargs)

    if name in ['notch', 'nf']:
        return notch(sr, **kwargs)

    if name in ['peak', 'pf']:
        return peak(sr, **kwargs)

    return biquad(sr, **kwargs)
