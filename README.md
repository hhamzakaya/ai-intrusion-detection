#  ai-intrusion-detection
AI-based real-time intrusion detection system using Python and NS-3

# AI-Based Intrusion Detection System

This project implements a Python-based machine learning system to detect network intrusions. It includes modules for data preprocessing, model training, hyperparameter tuning, evaluation, and visualization.

## Features

- Preprocessing of network log data
- Model training and validation with scikit-learn and LGBM
- Hyperparameter tuning for multiple algorithms
- Generation of alerts and performance metrics
- Confusion matrix and ROC curve visualization
- Designed for use with datasets such as UNSW-NB15

## Project Structure

- `data_utils.py`: Data loading and feature extraction functions
- `preprocessing.py`: Data cleaning and encoding logic
- `model.py`: Base model building and saving logic
- `model_validation.py`: Evaluation metrics and cross-validation
- `hyperparameter_tuning*.py`: Tuning scripts for various models
- `log_dashboard.py`: Alert dashboard with visualization (Streamlit or matplotlib)
- `alerts.csv`: Sample generated alerts with predictions
- `stage1_*.png`: Confusion matrix and ROC visualizations
- `stage1_binary_pipe.joblib`: Trained ML pipeline (excluded from GitHub if over 100MB)


