# Alterable biquad filters

![language](https://img.shields.io/badge/languages-C%2B%2B%20Python-blue)
![license](https://img.shields.io/github/license/jurihock/biquad?color=green)
![pypi](https://img.shields.io/pypi/v/biquad?color=gold)

This is a collection of [digital biquad filters](https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html) whose parameters `f` (frequency in *Hz*), `g` (gain in *dB*) and `q` (quality) can be varied at runtime. Following [DF1](https://ccrma.stanford.edu/~jos/fp/Direct_Form_I.html) filter implementations are available:

- Allpass
- Bandpass
- Highpass
- Lowpass
- Highshelf
- Lowshelf
- Notch
- Peak

## Basic usage

Filter with static configuration:

```python
import biquad
import numpy as np

# load audio samples somehow
x, sr = np.zeros(...), 44100

# create a filter of your choice
f = biquad.bandpass(sr, f=sr/4, q=1)

# process all audio samples
y = f(x)
```

Filter with dynamic configuration:

```python
import biquad
import numpy as np

# load audio samples somehow
x, sr = np.zeros(...), 44100

# create a filter of your choice
f = biquad.bandpass(sr)

# create parameter modifications as you like
myf = np.linspace(1, sr/4, len(x))
myq = np.linspace(2,  1/2, len(x))

# process all audio samples
y = f(x, f=myf, q=myq)
```

Keep in mind:

- All filters have a default value for the persistent parameters `g` and `q`, which is set in the particular `__init__` method.
- Parameter `f` must be set either in the `__init__` or in the `__call__` method.
- The optional instantaneous parameters `f`, `g` and `q`, if specified in the `__call__` method, override the persistent ones. 

## References

1. <span id="1">[Cookbook formulae for audio EQ biquad filter coefficients](https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html) by Robert Bristow-Johnson</span>
2. <span id="2">[Introduction to Digital Filters with Audio Applications](https://ccrma.stanford.edu/~jos/filters/filters.html) by Julius O. Smith III</span>

## License

[github.com/jurihock/biquad](https://github.com/jurihock/biquad) is licensed under the terms of the MIT license.
For details please refer to the accompanying [LICENSE](https://github.com/jurihock/biquad/raw/main/LICENSE) file distributed with it.
