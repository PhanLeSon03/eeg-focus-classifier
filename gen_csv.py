import scipy.io
import pandas as pd
import os

dataset_folder = '/home/jupyter/.cache/kagglehub/datasets/inancigdem/eeg-data-for-mental-attention-state-detection/versions/1/EEG Data'
files = sorted(os.listdir(dataset_folder))
all_dfs = []
channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
useful_channels = ['F7','F3','P7','O1','O2','P8','AF4']

fs = 128
n_subjects = 5
mkpt1 = int(fs*10*60)
mkpt2 = int(fs*20*60)


for file in files:
    print(file)
    mat_data = scipy.io.loadmat(os.path.join(dataset_folder,file))

    eeg_data = mat_data['o']['data'][0][0][:, 3:18]

    # Tạo danh sách tên kênh theo đúng thứ tự
    channel_names = [
        'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2',
        'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4', 'state'
    ]

    # Chuyển dữ liệu thành DataFrame
    df = pd.DataFrame(eeg_data, columns=channel_names)
  
    df['state'] = df['state'].astype(str)
    df.loc[:mkpt1, 'state'] = 'focused'
    df.loc[mkpt1:mkpt2, 'state'] = 'unfocused'
    df.loc[mkpt2:, 'state'] = 'drowsed'
    df['filename'] = file

    all_dfs.append(df)

final_df = pd.concat(all_dfs, ignore_index=True)
final_df.to_csv('preprocessed_eeg_data.csv', index=False)
print("✅ Saved preprocessed_eeg_data.csv successfully.")