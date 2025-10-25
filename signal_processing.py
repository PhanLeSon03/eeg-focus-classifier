import sys
import copy
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt

class WindowIIRNotchFilter():
    '''
    Class for real time IIR notch filtering.
    '''

    def __init__(self, w0, Q, fs):
        '''
        Arguments:
            w0: Center frequency of notch filter.
            Q: Quality factory of notch filter.
            fs: Sampling rate of signal that filtering will be performed on.
        '''
        self.w0 = w0
        self.Q = Q
        self.fs = fs
        self.initialize_filter_params()

    def initialize_filter_params(self):
        '''
        Initialize IIR notch filter parameters.
        '''
        self.b, self.a = scipy.signal.iirnotch(self.w0, self.Q, self.fs)
        self.z = scipy.signal.lfilter_zi(self.b, self.a)

    def filter_data(self, x):
        '''
        Apply notch filter to signal sample x.

        input:
            x: Window of signal data
        output:
            result: Filtered signal data
        '''
        result, zf = scipy.signal.lfilter(self.b, self.a, x, -1, self.z)
        self.z = zf
        return np.array(result)


class DCBlockingFilter:
    '''
    Class for window-based time DC Blocking filtering.
    '''

    def __init__(self, alpha=0.99):
        '''
        Arguments:
            alpha: Adaptation time-constant for DC drift
        '''
        self.alpha = alpha
        self.initialize_filter_params()

    def initialize_filter_params(self):
        '''
        Initialize filter parameters.
        '''
        self.b = [1, -1]
        self.a = [1, -1 * self.alpha]
        self.zi = scipy.signal.lfilter_zi(self.b, self.a)

    def filter_data(self, x):
        '''
        Apply filter to signal sample x.

        input:
            x: Window of signal data
        output:
            result: Filtered signal data
        '''
        result, zf = scipy.signal.lfilter(self.b, self.a, x, zi=self.zi)
        self.zi = zf
        return np.array(result)


class WindowFilter():
    '''
    Sliding window filtering class for de-noising slow wave data in deep sleep epochs.
    '''

    def __init__(self, filters):
        '''
        Arguments:
            filters: list of RealTime filter objects
        '''
        self.filters = filters

    def initialize_filter_params(self):
        '''
        Initializes RealTime filter object parameters.
        '''
        for filt in self.filters:
            filt.initialize_filter_params()

    def filter_data(self, x):
        '''
        Apply RealTime filters to signal sample x.

        input:
            x: Window of signal data
        output:
            result: Filtered signal data
        '''
        for filt in self.filters:
            x = filt.filter_data(x)
        return x


class WindowButterBandpassFilter():
    '''
    Class for real time Butterworth Bandpass filtering.
    '''

    def __init__(self, order, low, high, fs):
        '''
        Arguments:
            order: Bandpass filter order.
            low: Lower cutoff frequency (Hz).
            high: Higher cutoff frequency (Hz).
            fs: Sampling rate of signal that filtering will be performed on.
        '''
        self.order = order
        self.low = low
        self.high = high
        self.fs = fs
        self.initialize_filter_params()

    def initialize_filter_params(self):
        '''
        Initialize filter parameters.
        '''
        self.b, self.a = scipy.signal.butter(self.order, [self.low, self.high], btype='band', fs=self.fs)
        self.z = scipy.signal.lfilter_zi(self.b, self.a)

    def filter_data(self, x):
        '''
        Apply bandpass filter to signal sample x.

        input:
            x: Window of signal data
        output:
            result: Filtered signal data
        '''
        x = np.reshape(x, (-1,))
        result, zf = scipy.signal.lfilter(self.b, self.a, x, zi=self.z)
        self.z = zf
        return np.array(result)


if __name__ == "__main__":
    fs = 244
    dc_filter = DCBlockingFilter(alpha=0.99)
    notch_60 = WindowIIRNotchFilter(60, 12, fs)
    notch_50 = WindowIIRNotchFilter(50, 5, fs)
    notch_25 = WindowIIRNotchFilter(25, 10, fs)
    bandpass_filter = WindowButterBandpassFilter(3, 1, 35, fs)

    eeg_filter = WindowFilter([dc_filter, notch_60, notch_50, notch_25, bandpass_filter])


    duration = 30  # seconds
    total_samples = fs * duration

    t = np.arange(total_samples) / fs
    raw_signal = ( 300 +
            20 * np.sin(2 * np.pi * 1 * t) +  # 1 Hz delta wave
            10 * np.sin(2 * np.pi * 10 * t) +  # 10 Hz alpha wave
            10 * np.sin(2 * np.pi * 60 * t) +  # 60 Hz noise
            20 * np.random.randn(total_samples)  # random noise
    )

    window_len = fs
    filtered_signal = []

    for i in range(0, total_samples, window_len):
        window = raw_signal[i:i + window_len]
        if len(window) < window_len:
            break  # skip incomplete window at the end
        filtered_window = eeg_filter.filter_data(window)
        filtered_signal.extend(filtered_window)

    filtered_signal = np.array(filtered_signal)

    # Plot comparison
    plt.figure(figsize=(12, 5))
    plt.plot(t, raw_signal, label='Raw Signal', alpha=0.5)
    plt.plot(t[:len(filtered_signal)], filtered_signal, label='Filtered Signal', linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title("EEG Signal Filtering Example")
    plt.tight_layout()
    plt.show()

