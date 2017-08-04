# Keep this here because making a filter is never as simple as I would expect
from scipy.signal import freqz, lfilter, butter
from scipy.signal import filtfilt as ff
import matplotlib.pyplot as plt
import numpy as np

# http://stackoverflow.com/questions/12093594/
# how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter


def butter_bandpass(lowcut, highcut, Fs, order=5):
    nyq = 0.5 * Fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(
        data, lowcut, highcut, Fs, order=5, plot_filter=False, filtfilt=True,
        axis=-1):
    b, a = butter_bandpass(lowcut, highcut, Fs, order=order)
    y = ff(b, a, data, axis) if filtfilt else lfilter(b, a, data, axis)
    if plot_filter:
        filter_plot(b, a, Fs)
    return y


def butter_highpass(highcut, Fs, order=5):
    nyq = 0.5 * Fs
    high = highcut / nyq
    b, a = butter(order, high, btype='highpass')
    return b, a


def butter_highpass_filter(
        data, highcut, Fs, order=5, plot_filter=False, filtfilt=True, axis=-1):
    b, a = butter_highpass(highcut, Fs, order=order)
    y = ff(b, a, data, axis) if filtfilt else lfilter(b, a, data, axis)
    if plot_filter:
        filter_plot(b, a, Fs)
    return y


def butter_lowpass(lowcut, Fs, order=5):
    nyq = 0.5 * Fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='lowpass')
    return b, a


def butter_lowpass_filter(
        data, lowcut, Fs, order=5, plot_filter=False, filtfilt=True, axis=-1):
    b, a = butter_lowpass(lowcut, Fs, order=order)
    y = ff(b, a, data, axis) if filtfilt else lfilter(b, a, data, axis)
    if plot_filter:
        filter_plot(b, a, Fs)
    return y


def filter_plot(b, a, Fs):
    w, h = freqz(b, a)
    fig, ax = plt.subplots()
    ax.plot((Fs*0.5/np.pi)*w, abs(h))
