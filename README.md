# EEG Classification with Deep Learning, SMOTE & Explainability

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
[![Python Version](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)](https://www.python.org/)

---

## 📄 Project Overview

This repository contains code and documentation for training a Convolutional Neural Network (CNN) to classify EEG (Electroencephalography) signals into different mental/emotional states. Key aspects:

- **Deep Learning**: A PyTorch-based CNN architecture.  
- **Imbalanced Data Handling**: SMOTE oversampling on the training set.  
- **Explainability**: SHAP analysis to interpret model decisions.  

---

## 🔍 Table of Contents

- [Features](#-features)  
- [Dataset & Inputs](#-dataset--inputs)  
- [Preprocessing](#-preprocessing)  
- [Model Architecture](#-model-architecture)  
- [Training & Results](#-training--results)  
- [Explainable AI (XAI)](#-explainable-ai-xai)  
- [Usage](#-usage)  
- [Directory Structure](#-directory-structure)  
- [Requirements](#-requirements)  
- [License](#-license)  
- [Contributing](#-contributing)  

---

## 🚀 Features

- **CNN-based classifier** for raw EEG feature vectors.  
- **SMOTE balancing** to remedy class imbalance.  
- **Validation tracking**: final-epoch vs. best-epoch accuracy comparisons.  
- **SHAP explainability**: feature importance visualization.  

---

## 📥 Dataset & Inputs

- **Raw EEG features** were extracted into a tabular CSV (`data/eeg_features.csv`).  
- Each row corresponds to one time-window, with columns for spectral power bands and channel statistics.  
- The `label` column indicates the target mental/emotional state.

---

## 🧹 Preprocessing

1. **Missing Values**  
   - Imputed via median strategy.  
2. **Scaling**  
   - StandardScaler on all numeric features.  
3. **Train/Test Split**  
   - Stratified 80/20 split.  
4. **SMOTE**  
   - Applied to **training** data only to synthetically balance minority classes.

---

## 🏗️ Model Architecture

- **Convolutional Layers**  
  - Three 1D convolutional blocks (`Conv1d → ReLU → BatchNorm → MaxPool`).  
- **Fully Connected Layers**  
  - Two dense layers with dropout for regularization.  
- **Output**  
  - Softmax over `N` classes.  

Check `models/cnn.py` for full implementation.

---

## 📊 Training & Results

- **Epochs**: 200  
- **Before SMOTE**  
  - Final Val. Accuracy: 85.71 %  
  - Best Val. Accuracy: 91.07 %  
- **After SMOTE**  
  - Final Val. Accuracy: 81.43 %  
  - Best Val. Accuracy: 97.47 %  

> See the notebook for loss/accuracy curves and the full evaluation table.

---

## 🧠 Explainable AI (XAI)

We used **SHAP** to interpret the CNN’s predictions:

- Generated SHAP values for test samples.  
- Produced summary plots showing which EEG channels and frequency bands contribute most.  



