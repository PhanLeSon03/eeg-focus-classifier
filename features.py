from scipy.signal import welch
import random
import numpy as np
import antropy as ant
from scipy.fft import fft, fftfreq


def simpson(y, x=None, dx=1.0):
    """
    Numerical integration using Simpson's rule.
    If the number of points is even, the last interval is handled using the trapezoidal rule.

    Parameters:
    - y: Array of function values.
    - x: Array of sample points (optional).
    - dx: Spacing between points (used if x is None).

    Returns:
    - Integral approximation.
    """
    y = np.asarray(y)
    n = len(y)
    if n < 2:
        raise ValueError("At least 2 points are required.")
    
    if x is None:
        x = np.arange(n) * dx
    else:
        x = np.asarray(x)
        if len(x) != n:
            raise ValueError("x and y must have the same length.")

    if n % 2 == 1:
        # Odd number of points: apply Simpson's rule directly
        h = (x[1:] - x[:-1]).mean()
        result = y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2])
        result *= h / 3.0
    else:
        # Even number of points: apply Simpson's rule to first n-1, trapezoid to last interval
        result = simpson(y[:-1], x[:-1])  # recursive call on n-1 points
        trap = 0.5 * (x[-1] - x[-2]) * (y[-1] + y[-2])
        result += trap

    return result


def compute_z_ratio(datum, fs):
    '''
    Computes the z-ratio of a given signal segment.

    input:
        datum: segment of biopotential signal data
        fs: sampling rate of datum
    output:
        z-ratio, a measure of relative slow wave activity in the EEG signal
    '''
    fft_datum = np.abs(fft(datum))
    freqs = fftfreq(len(datum),1/fs)
    indice = np.bitwise_and(freqs<=(20), freqs>=0.5)
    fft_datum = fft_datum[indice]
    freqs = freqs[indice]
    total_pow = simpson(fft_datum,freqs)

    slow_indice = np.bitwise_and(freqs<=8, freqs>=0.5)
    slow_power = simpson(fft_datum[slow_indice],freqs[slow_indice])/(total_pow +1e-10)

    fast_indice = np.bitwise_and(freqs<=20, freqs>=8)
    fast_power = simpson(fft_datum[fast_indice],freqs[fast_indice])/(total_pow +1e-10)

    return (slow_power-fast_power)/(slow_power+fast_power +1e-10), slow_power/(fast_power +1e-10)

def extract_welch_features(window_data, fs=128, nperseg=256):
    # Welch PSD Calculation
    nperseg = min(len(window_data), nperseg)
    freqs, psd = welch(window_data, fs=fs, nperseg=nperseg)
    total_power = np.sum(psd)

    # Frequency bands
    bands = {'delta': (0.5, 4),
             'theta': (4, 8),
             'alpha': (8, 13),
             'beta':  (13, 30),
             'gamma': (30, 50)}

    # Band power calculations
    band_power = {}
    for band, (low, high) in bands.items():
        idx = (freqs >= low) & (freqs < high)
        band_power[band] = np.sum(psd[idx])
        
    band_power_merge ={
    'delta_theta':band_power['delta'] + band_power['theta'],
    'alpha': band_power['alpha'],
    'beta_gamma':band_power['beta'] + band_power['gamma'],
    }

    # Normalized power
    norm_band_power = {b: p/(total_power + 1e-10)
    for b, p in band_power.items()}

    # Ratios
    ratios = {
        'theta_alpha': band_power['theta']/(band_power['alpha'] + 1e-10),
        'theta_beta': band_power['theta']/(band_power['beta'] + 1e-10),
        'beta_alpha': band_power['beta']/(band_power['alpha'] + 1e-10),
        'beta_gamma': band_power['beta']/(band_power['gamma'] + 1e-10),
        'alpha_gamma': band_power['alpha']/(band_power['gamma'] + 1e-10)
    }


    spec_entropy = ant.spectral_entropy(window_data, sf=fs,
                                        method='welch', normalize=True)

    if np.isnan(spec_entropy):
        print("Warning: Input signal contains NaN or inf values.")
        spec_entropy = 1e10

    statistical_features = [np.std(window_data), np.max(window_data) - np.min(window_data)]

    ret0, ret1 = compute_z_ratio(window_data, fs)
    z_ratio_features = [ret0, ret1]


    # Concate all values
    features = np.array([
        *band_power_merge.values(),     # 3 features
        *norm_band_power.values(),      # 5 features
        *ratios.values(),               # 5 features
        *statistical_features,          # 3 features
        *z_ratio_features,              # 1 feature
        spec_entropy                    # 1 feature
    ])
    return features