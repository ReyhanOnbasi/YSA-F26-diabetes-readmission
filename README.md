# Predicting 30-Day Hospital Readmission in Patients with Diabetes

Term project for *Artificial Neural Networks (F26)*.

Binary prediction of readmission within 30 days on the UCI *Diabetes 130-US Hospitals for Years 1999–2008* dataset.
Six algorithm families are compared – Logistic Regression, k-NN, Gaussian Naive Bayes, Random Forest, XGBoost and
artificial neural networks (a scikit-learn MLP and a Keras network with categorical embeddings) – using a
patient-grouped stratified 60/20/20 train/validation/test split and leakage-free pipelines.

- Hyper-parameters: patient-grouped 3-fold cross-validation inside the training set (92 configurations)
- Validation set: model comparison and decision-threshold selection
- Test set: single final evaluation with patient-level bootstrap confidence intervals
- Additional analyses: neural architecture experiments and overfitting curves, patient-grouped 5-fold
  cross-validation, class weighting vs SMOTE (train-only, inside an imbalanced-learn pipeline),
  encounter- vs patient-level splitting, learning curves, calibration, permutation importance, subgroup analysis

## Files
- `YSA_F26_readmission.ipynb` – notebook with all outputs (run on Kaggle Notebooks, 4 CPU cores, ~1.5 h)
- `figures/` – all figures at 600 dpi
- `results/` – all result tables (CSV)

## Data
Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). Diabetes 130-US Hospitals for Years 1999–2008.
UCI Machine Learning Repository. https://doi.org/10.24432/C5230J
The notebook downloads the data automatically.

## Requirements
Python 3.10+, scikit-learn >= 1.4, xgboost, tensorflow/keras, imbalanced-learn, pandas, numpy, scipy,
matplotlib, seaborn (all preinstalled on Kaggle and Google Colab).
