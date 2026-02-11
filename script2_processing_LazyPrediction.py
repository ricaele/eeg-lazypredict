"""
Comprehensive EEG and Biosignal Classification Pipeline (Post-Cleaning)
Includes data loading, power spectral density (PSD) computation for GSR/ECGchannels,
Riemannian feature extraction for EEG, signal normalization, featureconcatenation, and classification.
Supports LazyClassifier benchmarking with combined EEG and peripheralphysiological signals.
Author: C. Ricaele
Date: 09/06/25

Requirements:
- mne
- numpy
- scipy
- scikit-learn
- pyriemann
- lazypredict

Instructions:
- Set 'base_path' to the directory containing participant folders.
- Ensure each folder contains an 'epochs_clean.fif' file.
- Modify participant identifiers and event labels to match your dataset.
- Output will be saved as a CSV file summarizing classification modelperformance.

"""


import os
import numpy as np
import mne
from scipy.signal import welch
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pyriemann.estimation import ERPCovariances
from pyriemann.tangentspace import TangentSpace
from lazypredict.Supervised import LazyClassifier
# ================================
# USER-SPECIFIC PATHS AND VARIABLES
# ================================
# Base folder where EEG folders for each participant are located
base_path = "your/base/folder" # <-- Modify this path
# List of participant folder names
participant_list = ["P01", "P02", "P03"] # <-- Modify participant identifiersas needed
# Output CSV file path
output_file = "path/to/save/results.csv" # <-- Change to your desired outputlocation
# Sampling frequency of your data
fs = 512 # <-- Set your data sampling rate
# ================================
# DATA LOADING AND PSD COMPUTATION
# ================================
X_combined_EEG = []
X_combined_Sensors = []
y_combined = []
for participant in participant_list:
file_path = os.path.join(base_path, participant, 'epochs_clean.fif')
# Load EEG data using MNE
if os.path.exists(file_path):
epochs = mne.read_epochs(file_path)
# Determine EEG channel count
n_eeg_channels = len(epochs.copy().pick_types(eeg=True).
↪info['ch_names'])
print(f'Number of EEG channels: {n_eeg_channels}')
# Print channel names and recording metadata
print(epochs.ch_names)
print(epochs.info)
# Select emotional conditions of interest
conditions = ['medo', 'felicidade', 'raiva', 'tristeza', 'neutro'] #<-- Adapt if needed
epochs = epochs[conditions]
# Get sensor data (excluding EEG)
data = epochs.get_data()[:, n_eeg_channels:, :]
n_epochs, n_channels, n_samples = data.shape
nperseg = 256
n_freqs_desired = 513
# Initialize array to store interpolated PSDs
psd = np.zeros((n_epochs, n_channels, n_freqs_desired))
# Compute PSD for each epoch and channel using Welch method
for epoch in range(n_epochs):
for channel in range(n_channels):
f, Pxx = welch(data[epoch, channel, :], fs=fs, nperseg=nperseg)
interp_func = interp1d(f, Pxx, kind='linear',fill_value='extrapolate')
f_new = np.linspace(f[0], f[-1], n_freqs_desired)
psd[epoch, channel, :] = interp_func(f_new)
# Replace sensor signal with computed PSD
epochs._data[:, n_eeg_channels:, :] = psd
# Split into EEG and sensor arrays
X = epochs.get_data()
y = epochs.events[:,-1]
X_eeg = X[:, :n_eeg_channels, :]
X_sensors = X[:, n_eeg_channels:, :]
X_combined_EEG.append(X_eeg)
X_combined_Sensors.append(X_sensors)
y_combined.append(y)
# ================================
# EEG PADDING (STANDARDIZE TO 17 CHANNELS)
# ================================
X_combined_EEG_padded = []
for data in X_combined_EEG:
if data.shape[1] < 17:
padding_channels = 17- data.shape[1]
padding = np.zeros((data.shape[0], padding_channels, data.shape[2]))
data = np.concatenate((data, padding), axis=1)
X_combined_EEG_padded.append(data)

# Convert lists to NumPy arrays
X_combined_EEG = np.concatenate(X_combined_EEG_padded, axis=0)
X_combined_Sensors = np.concatenate(X_combined_Sensors, axis=0)
y_combined = np.concatenate(y_combined, axis=0)
# ================================
# TRAIN-TEST SPLIT
# ================================
X_trainE, X_testE, y_trainE, y_testE = train_test_split(
X_combined_EEG, y_combined, test_size=0.2, random_state=42)
X_trainS, X_testS, y_trainS, y_testS = train_test_split(
X_combined_Sensors, y_combined, test_size=0.2, random_state=42)
# ================================
# PIPELINES: EEG (Riemann) AND SENSOR (Flattened)
# ================================
pipelineEEG = make_pipeline(
ERPCovariances(estimator='lwf'),
TangentSpace(metric="riemann"),
StandardScaler()
)
pipelineSENSORS = make_pipeline(StandardScaler())
# Fit-transform EEG pipeline
X_trainE_transformed = pipelineEEG.fit_transform(X_trainE, y_trainE)
X_testE_transformed = pipelineEEG.transform(X_testE)
# Flatten sensor data and scale
X_trainS_flat = X_trainS.reshape(X_trainS.shape[0],-1)
X_testS_flat = X_testS.reshape(X_testS.shape[0],-1)
X_trainS_transformed = pipelineSENSORS.fit_transform(X_trainS_flat, y_trainS)
X_testS_transformed = pipelineSENSORS.transform(X_testS_flat)
# Concatenate EEG and sensor features
X_train_final = np.concatenate((X_trainE_transformed, X_trainS_transformed),axis=1)
X_test_final = np.concatenate((X_testE_transformed, X_testS_transformed),axis=1)
# ================================
# CLASSIFICATION USING LazyClassifier
# ================================
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train_final, X_test_final, y_trainE, y_testE)
# ================================
# SAVE RESULTS TO CSV
# ================================
models.to_csv(output_file, index=False)
print(f'Classification results saved to: {output_file}')
