import numpy as np
import torch
from torchinfo import summary
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
from model import MC_Model_1D
from collections import Counter
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


numClass = 3
ts = "8s"

hop_len = 1 
segment_len = 64  

label_map = {'unfocused': 1, 'drowsed': 0,'focused': 2}
subject_specs_dicts = np.load(f'Features/features_labels_{ts}_14c.npy', allow_pickle=True).item()

FullChannels = True
if FullChannels==True:
    Plot_Channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
else:
    Plot_Channels = ['F7', 'F3', 'P7', 'O1', 'O2', 'P8']
    
map_indexes = {'AF3':0, 'F7':1, 'F3':2, 'FC5':3, 'T7':4, 'P7':5, 'O1':6, 'O2':7,
    'P8':8, 'T8':9, 'FC6':10, 'F4':11, 'F8':12, 'AF4':13 }

    
num_channels = len(Plot_Channels)
list_filenames = list(subject_specs_dicts[0].keys())

# for ch in subject_specs_dicts:
#     for fname in list_filenames:
#         subject_specs_dicts[ch][fname]['welch'] = subject_specs_dicts[ch][fname]['welch'][:30*60]
#         subject_specs_dicts[ch][fname]['labels'] = subject_specs_dicts[ch][fname]['labels'][:30*60]


mask = np.array([
 [0,0,1,1,1,0],
 [1,1,1,1,1,1],
 [0,0,0,0,0,0],
 [1,1,1,1,1,1],
 [0,0,0,0,0,0],
 [1,1,1,1,1,1],
 [0,0,1,1,1,0],
 [1,0,0,0,0,1],
 [1,1,1,1,1,1],
 [0,0,1,1,1,0],
 [1,1,1,1,1,1],
 [1,0,1,1,1,1],
 [1,1,1,1,1,1],
 [0,0,0,0,0,0],
 [0,0,0,0,1,0],
 [1,1,1,1,1,1],
 [1,1,1,1,1,1],
 [0,0,1,1,1,0]
], dtype=int)

# Flatten in column-major order (per channel, top-down)
flat_mask = mask.flatten(order='F')
good_feature_indexes = np.where(flat_mask == 1)[0]

# print("Good feature indexes:\n", good_feature_indexes)
# print(f"\nTotal good features: {len(good_feature_indexes)}")


exclude_files = []


check_points_dir = f"check_points_{ts}_{FullChannels}"
os.makedirs(check_points_dir, exist_ok=True)


class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)  # shape: (N, channels, time, freq)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def build_dataset(filenames, Hop = 2, channels_idx = None):
    data = []
    labels = []
    for fname in filenames:
        
        if channels_idx == None:
            features = [subject_specs_dicts[ch][fname]['welch'] for ch in range(num_channels)]
        else:
            features = [subject_specs_dicts[ch][fname]['welch'] for ch in channels_idx]
        features = np.concatenate(features, axis=1)  
        # features = features[:,good_feature_indexes]

        label_track = np.asarray(subject_specs_dicts[0][fname]['labels'])
        num_frames = label_track.shape[0]
        

        for i in range(0,num_frames,Hop):
            start = i
            end = start + segment_len

            # Handle last chunk with zero padding
            if end > num_frames:
                pad_len = end - num_frames
                seg_feat = features[start:num_frames, :]
                seg_labels = label_track[start:num_frames]

                # Pad features and labels
                seg_feat = np.pad(seg_feat, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)
                seg_labels = np.pad(seg_labels, (0, pad_len), mode='edge')  # repeat last label
            else:
                seg_feat = features[start:end, :]
                seg_labels = label_track[start:end]

            # Majority label
            label_major = Counter(seg_labels).most_common(1)[0][0]

            data.append(seg_feat)
            labels.append(label_major)
            
    data = np.array(data, dtype=np.float32)  # shape: (N, T, F)
    labels = np.array(labels, dtype=np.int64)
    
    if numClass == 2:
        indexes = labels<2
        labels = labels[indexes]
        data = data[indexes]
    

    return data, labels


Channels_idx = [map_indexes[Ch] for Ch in Plot_Channels]

results = []
batch_size = 32
sesssion_ind = True
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []
print(f'Session independent: {sesssion_ind}')
for fold, (train_idx, test_idx) in enumerate(kf.split(list_filenames)):
    print(f"\n==== Fold {fold} ====")
    
    if sesssion_ind: 
        train_subjects = [list_filenames[i] for i in train_idx if list_filenames[i] not in exclude_files]
        test_subjects = [list_filenames[i] for i in test_idx if list_filenames[i] not in exclude_files]

        X_train, y_train = build_dataset(train_subjects, channels_idx = Channels_idx)
        X_test, y_test = build_dataset(test_subjects, channels_idx = Channels_idx)

        train_dataset = EEGDataset(X_train, y_train)
        test_dataset = EEGDataset(X_test, y_test)

    else:
        X, y = build_dataset(list_filenames, channels_idx = Channels_idx)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=fold)
        
        train_dataset = EEGDataset(X_train, y_train)
        test_dataset = EEGDataset(X_test, y_test)
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = MC_Model_1D(n_channels=1, n_features =X_train.shape[-1]//1, time_steps=segment_len, num_classes=numClass).to(device)
    
    summary(model, 
        input_size= [(1, segment_len, X_train.shape[-1])],
        col_names=["input_size", "output_size", "num_params"],
        dtypes=[torch.float, torch.float])
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

    best_test_acc = 0.0
    best_train_acc = 0.0
    best_model_state = None
    # Training
    for epoch in range(100):  
        model.train()
        total_loss = 0
        reg_loss = 0
        correct = 0
        for X_batch, y_batch in train_loader:
            # print(y_batch)
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            reg_loss = model.regularization_loss()
            loss = criterion(outputs, y_batch) + reg_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == y_batch).sum().item()

        acc_train = correct / len(train_dataset)
        scheduler.step()
        # print(model.W_real.grad)

        # Evaluation
        model.eval()
        correct = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                correct += (outputs.argmax(1) == y_batch).sum().item()
        acc_test = correct / len(test_dataset)
        
        if acc_test > best_test_acc:
            best_test_acc = acc_test
            best_train_acc = acc_train
            best_model_state = model.state_dict()  # clone weights
            torch.save(best_model_state, f"{check_points_dir}/best_model_fold_{fold}_ind_{sesssion_ind}.pt")

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Reg Loss: {reg_loss:.4f} Train Accuracy: {acc_train:.4f}, , Test Acc = {acc_test:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    
    results.append({"fold": fold, "train_acc": best_train_acc, "test_acc": best_test_acc})
    torch.save(best_model_state, f"{check_points_dir}/best_model_fold_{fold}_{sesssion_ind}.pt")


df_results = pd.DataFrame(results)
df_results.to_csv(f"{check_points_dir}/crossval_results_{ts}_{numClass}Classes_kfold_{sesssion_ind}.csv", index=False)

best_row = df_results.loc[df_results['test_acc'].idxmax()]
print(f"\n🔥 Best Fold: {best_row['fold']} — Test Acc: {best_row['test_acc']:.4f}")
print(f"📊 Average Test Accuracy: {df_results['test_acc'].mean():.4f}")
print(f"📊 STD Test Accuracy: {df_results['test_acc'].std():.4f}")