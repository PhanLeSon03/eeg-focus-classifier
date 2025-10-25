# eeg-focus-classifier

This repository contains code and scripts for analyzing EEG signals and classifying mental states — **Focus**, **Unfocus**, and **Relaxation** — using both classical machine learning and deep neural network models.  
The framework supports multiple evaluation settings, including **session-dependent**, **session-independent**, and **subject-independent (LOSO)** protocols.

---

## Overview

Workflow:

1. Download and preprocess EEG datasets  
2. Extract statistical and spectral features  
3. Train and evaluate XGBoost and RNN-based models  
4. Analyze feature importance and model performance under different experimental protocols

---

### Download the dataset
```
python download_dataset.py
```

### Generate a single csv file from .mat files
```
python gen_csv.py
```

### Generate feature .npy for neural network model's training
```
python gen_features_npy.py
```

### Study the feature: `study_features_Emotiv.ipynb`


### Train neural network models
#### For session-independence and session-dependence 
set sesssion_ind = True/False for session-independence/session-dependence 

set FullChannels = True/False for full channels/ 6 channels
```
python train_nn_DRNN.py
```
#### For subject-independence

set FullChannels = True/False for full channels/ 6 channels
```
python train_nn_DRNN_LOSO.py
```

### Evaluate the model: `validate_RNN.ipynb`

### Train XGBoost models
set LOSO = TRUE for subject-independence 

set sesssion_ind = True/False for session-independence/session-dependence 

set FullChannels = True/False for full channels/ 6 channels
```    
python train_XGBoost.py
```



     

