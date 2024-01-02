# PDHIs_ML
the code for PDHIs analysis
This is the Python code for a clinical prediction model, fitted by Dr. Shu Chang from Huanhu Hospital in Tianjin City, after integrating data from the HIS (Hospital Information System).

Hardware requirements：The codes requires only a standard computer with enough RAM to support the in-memory operations.
Software requirements：(1)Operating System：Linux: Ubuntu 20.04. (2)Python Packages:Python Version: 3.8.13, XGBoost: 1.6.1, scikit-learn: 1.1.1, SHAP: 0.41.0, pandas: 1.4.2, numpy: 1.22.3, matplotlib: 3.5.2
License: This project is covered under the Apache 2.0 License.
#The correspondence between Python code files and the methods section of the paper is as follows: (1)model_screening.py executes the initial screening process of constructing models with 15 different machine learning algorithms for various input variables.
(2)PDHIs_code_fitting_and_calibration.py is dedicated to fitting the XGBoost models for the three selected input variables, selecting hyperparameters, and performing calibration.
(3)McNemar_review.py carries out the final model selection using McNemar's test with the Benjamini-Hochberg correction.
(4)sensitive_analysis_shap_interaction_plot.py is the code for conducting individual and global sensitivity analyses, and it also generates SHAP interaction values plots."
