 """
Comprehensive Preprocessing Pipeline for EEG, ECG, and GSR Data (BrainVision␣
↪format)
Includes data loading, filtering, ICA cleaning, normalization, event␣
↪extraction, epoching, and artifact rejection.
Author: C. Ricaele
Date: 09/06/25
Requirements: mne, numpy, pandas, scikit-learn, mne-icalabel, autoreject
"""
import os
import mne
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler as Scaler
from mne.preprocessing import ICA
from mne_icalabel import label_components
from autoreject import AutoReject
full_path = "path/your/data.vhdr"
# Set EEG montage
# Load and label channels
raw = mne.io.read_raw_brainvision(full_path, preload=True)
# Indicate the sensor types for each channel
raw.set_channel_types({
raw.ch_names[-2]: 'ecg', # ECG
raw.ch_names[-1]: 'gsr' # GSR
})
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage)
# Apply bandpass and notch filters
raw_filtered = raw.copy().filter(l_freq=0.5, h_freq=50.0) # Define frequency␣
↪band
raw_filtered.notch_filter(freqs=60) # Apply notch filter to remove line noise
raw_filtered.set_eeg_reference("average") # Set average reference
# ICA decomposition
# Define the number of components
ica = ICA(n_components=16, max_iter="auto", method="infomax", random_state=97,␣
↪fit_params=dict(extended=True))
ica.fit(raw_filtered)
# Label and exclude non-brain ICA components
ic_labels = label_components(raw_filtered, ica, method="iclabel")
exclude_idx = [idx for idx, label in enumerate(ic_labels["labels"]) if label␣
↪not in ["brain", "other"]]
ica.exclude = exclude_idx
reconst_raw = raw_filtered.copy()
ica.apply(reconst_raw)
# Normalize using z-score
raw_data = reconst_raw.get_data()
scaler = Scaler(with_mean=True, with_std=True)
normalized_data = scaler.fit_transform(raw_data.T).T
raw_copy = mne.io.RawArray(normalized_data, reconst_raw.info)
raw_copy.set_annotations(reconst_raw.annotations)
# Extract events from annotations
events, event_dict = mne.events_from_annotations(raw_copy)
mne.viz.plot_events(events=events, event_id=event_dict)
# Epoching: extract segments from-200 ms to +800 ms relative to events
epochs = mne.Epochs(
raw_copy,
events,
event_id=event_dict,
tmin=-0.2, # Define epoch start time
tmax=0.8, # Define epoch end time
preload=True
)
# Select only the desired conditions for cleaning
selected_conditions = ['fear', 'happiness', 'anger', 'sadness', 'neutral']
ar = AutoReject()
epochs_clean = ar.fit_transform(epochs[selected_conditions])
path = "path/your/data/"
# Save cleaned epochs
epochs_clean.save(os.path.join(path, "epochs_clean.fif"), overwrite=True)
