# Alterable biquad filters

![language](https://img.shields.io/badge/languages-C%2B%2B%20Python-blue)
![license](https://img.shields.io/github/license/jurihock/biquad?color=green)
![pypi](https://img.shields.io/pypi/v/biquad?color=gold)

This is a collection of digital biquad filters whose parameters `frequency` and `quality` can be varied at runtime. Following filter implementations are available:

- Allpass
- Bandpass
- Highpass
- Lowpass
- Highshelf
- Lowshelf
- Notch
- Peak

## Basic usage

Filter with persistent configuration:

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

## See also

[Cookbook formulae for audio equalizer biquad filter coefficients](https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html)
