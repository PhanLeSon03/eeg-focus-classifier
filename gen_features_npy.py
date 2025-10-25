import numpy as np
import pandas as pd
from scipy.signal import spectrogram, stft

from collections import Counter
import os

# from scipy.signal import welch
import antropy as ant
from scipy.fft import fft, fftfreq
from features import extract_welch_features
from signal_processing import DCBlockingFilter, WindowIIRNotchFilter, WindowButterBandpassFilter, WindowFilter

def create_filter_chain(fs):
    return WindowFilter([
        DCBlockingFilter(alpha=0.99),
        # WindowIIRNotchFilter(60, 10, fs),
        # WindowIIRNotchFilter(50, 10, fs),
    ])

def preprocess_dataset(file_path):
    """
    Load and preprocess EEG dataset.

    Parameters:
    - file_path: str, path to the CSV file.
    - row_limit: int or None, number of rows to limit for processing (None means no limit).

    Returns:
    - df: pandas DataFrame, the preprocessed DataFrame.
    """
    eeg_filter = create_filter_chain(fs)
    
    # Load the data and drop completely empty columns
    data = pd.read_csv(file_path)
    df = data.dropna(axis=1, how='all')


    # Remove rows with 'unfocused' state
    # df = df[df['state'] != 'unfocused']

    # Map 'state' to integers: 'focused' -> 1, 'drowsed' -> 0
    df['state'] = df['state'].map({'focused': 2, 'drowsed': 0,'unfocused':1})

    keep_columns = [
        'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2',
        'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4', 'state'
    ]
    # df['Fz'] = (df['AF3'] + df['AF4'])/2
    # df['Cz'] = (df['O1'] + df['O2'])/2
    
    # keep_columns = [
    #  'F3', 'AF4', 'P7', 'P8', 'state'
    # ]

    
    for col in keep_columns[:-1]:
        df[col] = eeg_filter.filter_data(df[col].to_numpy())
        
    if 'filename' in df.columns:
        keep_columns.append('filename')

    df = df[keep_columns]

    return df

# Parameters: Welch feature for the signal with lenth: window_size (4s) and hop (1s)
# channels = [ 'F3', 'AF4', 'P7', 'P8']
channels =  [
        'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2',
        'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
fs = 128
hop = fs

window_size = 8*fs

# Load EEG data
df = preprocess_dataset('preprocessed_eeg_data.csv')


# Initialize dictionary for each channel
subject_specs_dicts = {i: {} for i in range(len(channels))}

# Group by file
grouped = df.groupby('filename', sort=False)

for filename, file_df in grouped:
    labels = file_df['state'].values
    
    for ch_idx, ch in enumerate(channels):
        signal = file_df[ch].values
        
        # Compute labels for each time frame by majority vote
        frame_labels = []
        features = []
        num_steps = (len(signal) - window_size + hop)//fs
        for i in range(num_steps):
            start = i * hop
            end = start + window_size
              
            segment_labels = labels[end-hop:end]
            label = Counter(segment_labels).most_common(1)[0][0]
            frame_labels.append(label)

            window_df = signal[start:end]
            welch_ = extract_welch_features(window_df, fs=fs)
            features.append(welch_)  

        subject_specs_dicts[ch_idx][filename] = {
            'welch': np.array(features),
            'labels': frame_labels
        }
        print(f'{filename}, welch size {np.array(features).shape}, label size {np.array(frame_labels).shape}')

# Save result
os.makedirs('Features', exist_ok=True)
np.save(f'Features/features_labels_{window_size//fs}s_14c.npy', subject_specs_dicts)
print("✅ Saved dictionary with majority-voted per-frame labels and spectrograms.")