import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import random
from scipy.stats import skew, kurtosis

import matplotlib.pyplot as plt
from scipy.signal import spectrogram

import os
from signal_processing import DCBlockingFilter, WindowIIRNotchFilter, WindowButterBandpassFilter, WindowFilter
from features import extract_welch_features

from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from collections import Counter

import joblib
import gc
gc.collect()


def create_filter_chain(fs):
    return WindowFilter([
        DCBlockingFilter(alpha=0.99),
        # WindowIIRNotchFilter(60, 10, fs),
        # WindowIIRNotchFilter(50, 10, fs),
    ])
fs = 128
sampling_rate  =128
segment_len = 64  

keep_columns = [
    'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2',
    'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4', 'state'
]

def preprocess_dataset(file_path):
    """
    Load and preprocess EEG dataset.

    Parameters:
    - file_path: str, path to the CSV file.
    - row_limit: int or None, number of rows to limit for processing (None means no limit).

    Returns:
    - df: pandas DataFrame, the preprocessed DataFrame.
    """
    eeg_filter_AF3 = create_filter_chain(fs)
    eeg_filter_AF4 = create_filter_chain(fs)
    
    # Load the data and drop completely empty columns
    data = pd.read_csv(file_path)
    df = data.dropna(axis=1, how='all')


    # Remove rows with 'unfocused' state
    # df = df[df['state'] != 'unfocused']

    # Map 'state' to integers: 'focused' -> 1, 'drowsed' -> 0
    df['state'] = df['state'].map({'focused': 2, 'drowsed': 0,'unfocused':1})

    # keep_columns = [
    #     'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2',
    #     'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4', 'state'
    # ]
    # df['Fz'] = (df['AF3'] + df['AF4'])/2
    # df['Cz'] = (df['O1'] + df['O2'])/2

    
    for col in keep_columns[:-1]:
        df[col] = eeg_filter_AF3.filter_data(df[col].to_numpy())
        
    if 'filename' in df.columns:
        keep_columns.append('filename')

    df = df[keep_columns]

    return df

hop_len = sampling_rate 
# @title save_welch_data
def save_welch_data(df, window_size, sampling_rate):
    
    save_to_csv = {key: [] for key in ['filename', 'state']}
    # Remove 'filename' and 'state' columns once, avoid doing it repeatedly inside the loop
    signal_columns = [col for col in df.columns if col not in ['filename', 'state']]

    # Group the DataFrame by 'filename' to avoid filtering in each loop
    grouped = df.groupby('filename', sort=False)

    # Initialize lists for features and labels
    features = []
    labels = []

    for file, file_df in grouped:
        # Calculate the number of windows at the start
        # num_windows = len(file_df) // window_size
        # num_windows = min (len(file_df), sampling_rate*60*20) // window_size
        # numSamples = min (len(file_df), sampling_rate*60*20)
        numSamples = len(file_df)
        num_windows = (numSamples - window_size +hop_len)// hop_len
        # Iterate through windows and calculate features
        for i in range(num_windows):
            save_to_csv['filename'].append(file)

            window_df = file_df.iloc[i * hop_len:i*hop_len +  window_size]

            # Extract features from all signal columns
            channel_features = []
            for col in signal_columns:
              welch_ = extract_welch_features(window_df[col].values, fs=sampling_rate)
              welch_name = [f"welch_{name}" for name in range(len(welch_))]
              for idx, name in enumerate(welch_name):
                name_col = f"{name}_{col}"
                if name_col in save_to_csv:
                  save_to_csv[name_col].append(welch_[idx])
                else:
                  save_to_csv[name_col] = [welch_[idx]]


              channel_features.append(welch_)

            # Concatenate and append features
            features.append(np.concatenate(channel_features))
            
            numCount = [0] * 3
            numCount[1] = np.count_nonzero(window_df['state'] == 1)
            numCount[0] = np.count_nonzero(window_df['state'] == 0)
            numCount[2] = np.count_nonzero(window_df['state'] == 2)

            representative_value = np.argmax(numCount)
          
            # Append the corresponding label
            labels.append(representative_value)
            save_to_csv['state'].append(representative_value)

    return np.array(features), np.array(labels), save_to_csv


# csv_path = 'preprocessed_eeg_data.csv'

# df = preprocess_dataset(csv_path)

# print(df.shape)
# df.head()

# window_size = 8*sampling_rate
# _, _, save_to_csv = save_welch_data(df, window_size, sampling_rate)
# df_welch = pd.DataFrame(save_to_csv)
# df_welch = df_welch.sort_values(by=['state','filename']).reset_index(drop=True)
# df_welch.to_csv('welch_data_8s_14c.csv', index=False, header=True, encoding='utf-8')
# del df
# gc.collect()
def create_features_without_initial(df, frame_size=64, hop=2, pad_last=False, features_to_add =None):
    """
    Build fixed-length windows per file and return (X, y) ready for ML models.
    
    Args:
        df : DataFrame with columns [filename, state, <features...>]
        frame_size : number of sequential rows per window
        hop : stride between window starts (in rows)
        pad_last : if True, pad short groups to frame_size and keep one window

    Returns:
        X : (num_windows, frame_size * num_features) float32
        y : (num_windows,) int64
    """
    
    # df = (df.groupby(df.columns[0], group_keys=False).head(60*30))
    X, y = [], []
    grouped = df.groupby('filename', sort=False)

    for _, group in grouped:
        
        
        if features_to_add == None:
            feats = group.iloc[:, 2:].values  # shape: (N, F)
            labs = group['state'].values     # shape: (N,)
        else:
            feats = group[features_to_add].values  # shape: (N, F)
            labs = group['state'].values       # shape: (N,)
        
        N, F  = feats.shape

        if N < frame_size:
            continue

        for i in range(0, N - frame_size + 1, hop):
            win = feats[i:i + frame_size, :]                 # (frame_size, F)
            lab_win = labs[i:i + frame_size]
            X.append(win.reshape(-1))                        # flatten to 1-D
            y.append(int(Counter(lab_win).most_common(1)[0][0]))

    if not X:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)

    X = np.stack(X, axis=0).astype(np.float32)               # (num_windows, D)
    y = np.asarray(y, dtype=np.int64)
    return X, y


best_fold = 9
best_model = None
best_means = None
best_stds = None
max_init_duration_sec = 300

'''
https://www.kaggle.com/datasets/inancigdem/eeg-data-for-mental-attention-state-detection/discussion/145650
The first 7 experiments belong to Subject 1, the next 7 experiments belong to Subject 2, and go on. 
However, Subject 5 could not complete the last experiment. 
Therefore, there are 34 records in the dataset.
'''
df_welch = pd.read_csv("welch_data_8s_14c.csv")
save_dict = {key: [] for key in ["acc_train", "acc_test"]}


results = []
exclude_files = []
list_filenames = list(df_welch['filename'].unique())

# ---------------- LOSO cross-validation ----------------
# Assumes list_filenames is ordered by session 1..34
SUBJECT_RANGES = {
    1: (0, 7),    # sessions 1..7
    2: (7, 14),   # 8..14
    3: (14, 21),  # 15..21
    4: (21, 28),  # 22..28
    5: (28, 34),  # 29..34
}

LOSO = True
sesssion_ind = True
FullChannels = True

if FullChannels==True:
    Plot_Channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
else:
    Plot_Channels = ['F7', 'F3', 'P7', 'O1', 'O2', 'P8']
features_idx = [f"welch_{idx}_{Ch}" for Ch in Plot_Channels for idx in range(18)]



if LOSO:
    results = []
    for subj_id in range(1, 6):
        print(f"\n==== LOSO: Hold-out Subject {subj_id} ====")
        s, e = SUBJECT_RANGES[subj_id]


        test_subjects = list_filenames[s:e]
        train_subjects = list_filenames[:s] + list_filenames[e:]

        train_df = df_welch[df_welch['filename'].isin(train_subjects)]
        test_df  = df_welch[df_welch['filename'].isin(test_subjects)]


        X_train, y_train = create_features_without_initial(train_df, frame_size=segment_len, features_to_add= features_idx)
        X_test,  y_test  = create_features_without_initial(test_df, frame_size=segment_len, features_to_add= features_idx)
        gc.collect()
        
        
        counts = np.bincount(y_train)
        if (len(counts) > 2):
            print(f"N.o 0: {counts[0]}, N.o 1: {counts[1]}, N.o 2: {counts[2]}")
        else:
            print(f"N.o 0: {counts[0]}, N.o 1: {counts[1]}")


        assert set(np.unique(y_train)) == {0,1,2}, "Training data must include all classes."
        
        clf = xgb.XGBClassifier(
            n_estimators=25, max_depth=3, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob",  # use softprob if you need probabilities
            num_class=3
        )
        clf.fit(
            X_train, y_train,
         )


        # y_pred_train = clf.predict(X_train)
        y_proba = clf.predict_proba(X_train)
        y_pred_train = np.argmax(y_proba, axis=1)
        accuracy_train = accuracy_score(y_train, y_pred_train)
        save_dict["acc_train"].append(accuracy_train)

        # y_pred_test = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        y_pred_test = np.argmax(y_proba, axis=1)
        accuracy_test = accuracy_score(y_test, y_pred_test)
        save_dict["acc_test"].append(accuracy_test)

        print("*"*40)
        print(f"Leave out S{subj_id}: accuracy_train = {accuracy_train} --> accuracy_test = {accuracy_test}")
        
        del X_train, y_train, X_test, y_test, y_pred_train, y_pred_test
        gc.collect()
        
    results_df = pd.DataFrame(save_dict) # pd.DataFrame({'random_state': range(100), 'acc_train': accurac


    max_index = results_df["acc_test"].idxmax()
    acc_test =  results_df["acc_test"][max_index]
    acc_train = results_df["acc_train"][max_index]
    print(f"Train accuracy {acc_train}, Maximum test accuracy: {acc_test}")
    print("At state:", max_index)
    exit()
         

print(f'Session independent: {sesssion_ind}')
if sesssion_ind:
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    list_filenames = list(df_welch['filename'].unique())

    for fold, (train_idx, test_idx) in enumerate(kf.split(list_filenames)):
        print(f"\n==== Fold {fold} ====")
        train_subjects = [list_filenames[i] for i in train_idx if list_filenames[i] not in exclude_files]
        test_subjects  = [list_filenames[i] for i in test_idx  if list_filenames[i] not in exclude_files]

        train_df = df_welch[df_welch['filename'].isin(train_subjects)]
        test_df  = df_welch[df_welch['filename'].isin(test_subjects)]

        X_train, y_train = create_features_without_initial(train_df, frame_size=segment_len, features_to_add= features_idx)
        X_test,  y_test  = create_features_without_initial(test_df, frame_size=segment_len, features_to_add= features_idx)
        gc.collect()
        
        
        counts = np.bincount(y_train)
        if (len(counts) > 2):
            print(f"N.o 0: {counts[0]}, N.o 1: {counts[1]}, N.o 2: {counts[2]}")
        else:
            print(f"N.o 0: {counts[0]}, N.o 1: {counts[1]}")


        assert set(np.unique(y_train)) == {0,1,2}, "Training data must include all classes."

        clf = xgb.XGBClassifier(
            n_estimators=25, max_depth=3, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob",  # use softprob if you need probabilities
            num_class=3
        )
        clf.fit(
            X_train, y_train,
        )

        if fold == best_fold:
            best_model = clf

        # y_pred_train = clf.predict(X_train)
        y_proba = clf.predict_proba(X_train)
        y_pred_train = np.argmax(y_proba, axis=1)
        accuracy_train = accuracy_score(y_train, y_pred_train)
        save_dict["acc_train"].append(accuracy_train)

        # y_pred_test = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        y_pred_test = np.argmax(y_proba, axis=1)
        accuracy_test = accuracy_score(y_test, y_pred_test)
        save_dict["acc_test"].append(accuracy_test)

        print("*"*40)
        print(f"Fold {fold}: accuracy_train = {accuracy_train} --> accuracy_test = {accuracy_test}")
        del X_train, y_train, X_test, y_test, y_pred_train, y_pred_test
        gc.collect()
else:
    # Build the full pooled dataset once
    X, y = create_features_without_initial(df_welch, frame_size=segment_len, features_to_add= features_idx)
    X = np.asarray(X)
    y = np.asarray(y)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n==== Fold {fold} ====")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        print(len(train_idx))
        print(len(test_idx))
        
        counts = np.bincount(y_train)
        if (len(counts) > 2):
            print(f"N.o 0: {counts[0]}, N.o 1: {counts[1]}, N.o 2: {counts[2]}")
        else:
            print(f"N.o 0: {counts[0]}, N.o 1: {counts[1]}")


        assert set(np.unique(y_train)) == {0,1,2}, "Training data must include all classes."

        clf = xgb.XGBClassifier(
            n_estimators=25, max_depth=3, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob",  # use softprob if you need probabilities
            num_class=3
        )
        clf.fit(
            X_train, y_train,
        )

        if fold == best_fold:
            best_model = clf

        # y_pred_train = clf.predict(X_train)
        y_proba = clf.predict_proba(X_train)
        y_pred_train = np.argmax(y_proba, axis=1)
        accuracy_train = accuracy_score(y_train, y_pred_train)
        save_dict["acc_train"].append(accuracy_train)

        # y_pred_test = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        y_pred_test = np.argmax(y_proba, axis=1)
        accuracy_test = accuracy_score(y_test, y_pred_test)
        save_dict["acc_test"].append(accuracy_test)

        print("*"*40)
        print(f"Fold {fold}: accuracy_train = {accuracy_train} --> accuracy_test = {accuracy_test}")
        del X_train, y_train, X_test, y_test, y_pred_train, y_pred_test
        gc.collect()
  

results_df = pd.DataFrame(save_dict) # pd.DataFrame({'random_state': range(100), 'acc_train': accurac



max_index = results_df["acc_test"].idxmax()
acc_test =  results_df["acc_test"][max_index]
acc_train = results_df["acc_train"][max_index]
print(f"Train accuracy {acc_train}, Maximum test accuracy: {acc_test}")
print("At state:", max_index)
model_dir = 'models_v01'
model_path = os.path.join(model_dir, f"xgb_model_6c.pkl")

joblib.dump(best_model, model_path)