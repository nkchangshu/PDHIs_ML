
#Clinical Prediction Model for Acute Ischemic Stroke
This repository contains the Python code for feature selection, hyperparameter optimization, and calibration of a clinical prediction model. The model was developed by Dr. Chang Shu from Tianjin Huanhu Hospital, Tianjin City, and is based on data integrated from the Hospital Information System (HIS).

The model aims to predict acute ischemic stroke and analyze predictive factors using hematological indicators in elderly hypertensive patients post-transient ischemic attack. This work supports the findings presented in the article titled "Acute ischemic stroke prediction and predictive factors analysis using hematological indicators in elderly hypertensives post‑transient ischemic attack," published in Scientific Reports.

Hardware requirements：The codes requires only a standard computer with enough RAM to support the in-memory operations.
Software requirements：(1)Operating System：Linux: Ubuntu 20.04. (2)Python Packages:Python Version: 3.8.13, XGBoost: 1.6.1, scikit-learn: 1.1.1, SHAP: 0.41.0, pandas: 1.4.2, numpy: 1.22.3, matplotlib: 3.5.2, optuna: 3.0.2
License: This project is covered under the Apache 2.0 License.

#The correspondence between Python code files and the methods section of the paper is as follows: (1)model_screening.py executes the initial screening process of constructing models with 15 different machine learning algorithms for various input variables.
(2)PDHIs_code_fitting_and_calibration.py is dedicated to fitting the XGBoost models for the three selected input variables, selecting hyperparameters, and performing calibration."
