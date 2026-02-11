# eeg-lazypredict
🧠 EEG and Biosignal Classification Pipeline

This repository contains a comprehensive preprocessing and classification pipeline for EEG, ECG, and GSR signals using:

MNE-Python

ICA with ICLabel

AutoReject

Riemannian geometry (pyRiemann)

Welch PSD estimation

LazyPredict benchmarking

📂 Pipeline Overview
1️⃣ Preprocessing (preprocessing_pipeline.py)

BrainVision file loading

Channel type assignment (EEG, ECG, GSR)

Band-pass and notch filtering

Average referencing

ICA decomposition (Infomax)

ICLabel-based artifact removal

Z-score normalization

Event extraction

Epoching (-200ms to 800ms)

AutoReject artifact correction

Output: epochs_clean.fif

2️⃣ Classification (riemann_lazypredict_pipeline.py)

Load cleaned epochs

PSD computation (Welch) for ECG/GSR

Riemannian ERP covariance estimation

Tangent space projection

Feature concatenation

Train-test split

LazyClassifier benchmarking

Output: CSV with model performance

▶️ How to Run
pip install -r requirements.txt


Run preprocessing:

python preprocessing/preprocessing_pipeline.py


Run classification:

python classification/riemann_lazypredict_pipeline.py

📌 Data Availability

Data are not publicly available due to privacy restrictions but may be requested from the corresponding author.

Isso deixa seu repositório com padrão internacional.

🥇 ETAPA 5 — C
