# Keep this here because making a filter is never as simple as I would expect
from scipy.signal import freqz, lfilter, butter
import matplotlib.pyplot as plt
import numpy as np

# http://stackoverflow.com/questions/12093594/
# how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
# Note difference here is to not express cutoffs as fraction of Nyquist freq


def butter_bandpass(lowcut, highcut, Fs, order=5):
    b, a = butter(order, [lowcut, highcut], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, Fs, order=5, plot_filter=False):
    b, a = butter_bandpass(lowcut, highcut, Fs, order=order)
    y = lfilter(b, a, data)
    if plot_filter:
        filter_plot(b, a, Fs)
    return y


def butter_highpass(highcut, Fs, order=5):
    b, a = butter(order, highcut, btype='highpass')
    return b, a


def butter_highpass_filter(data, highcut, Fs, order=5, plot_filter=False):
    b, a = butter_highpass(highcut, Fs, order=order)
    y = lfilter(b, a, data)
    if plot_filter:
        filter_plot(b, a, Fs)
    return y


def butter_lowpass(lowcut, Fs, order=5):
    b, a = butter(order, lowcut, btype='lowpass')
    return b, a


def butter_lowpass_filter(data, lowcut, Fs, order=5, plot_filter=False):
    b, a = butter_lowpass(lowcut, Fs, order=order)
    y = lfilter(b, a, data)
    if plot_filter:
        filter_plot(b, a, Fs)
    return y


def filter_plot(b, a, Fs):
    w, h = freqz(b, a)
    fig, ax = plt.subplots()
    ax.plot((Fs*0.5/np.pi)*w, abs(h))
