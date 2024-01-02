#training set
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
import pickle
os.chdir(r'/home/changshu/python2022/huanhu_data/hypertension_final_code')
col_xcg = ['SIRI', 'HCT', 'RDW_CV', 'PLT', 'BAS_p', 'IG_p', 'EOS']
X_train = pd.read_csv('X_train.csv')[col_xcg]
y_train = pd.read_csv('y_train.csv')

with open('robust_scaler.pkl', 'rb') as file:
    scaler_loaded = pickle.load(file)

X_train = scaler_loaded.transform(X_train)
X_train = pd.DataFrame(X_train, columns=col_xcg).reset_index(drop=True)

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

#import pickle
Pkl_Filename = "save/SR1_xgb_xcg_model.pkl"  
with open(Pkl_Filename, 'rb') as file:  
    xgboost_final_xcg = pickle.load(file)
import shap
import matplotlib.pyplot as plt
# create a SHAP explainer object
explainer = shap.TreeExplainer(xgboost_final_xcg)

# calculate shap values for the entire training set
shap_values = explainer.shap_values(X_train)
########

# Plot SHAP for a single sample
shap.force_plot(explainer.expected_value, shap_values[0,:], X_train.iloc[0,:], matplotlib=True)
# Plot the dependency graph for interactions
shap.dependence_plot('SIRI', shap_values, X_train, interaction_index='PLT')
shap.dependence_plot('SIRI', shap_values, X_train, interaction_index='HCT')
shap.dependence_plot('RDW_CV', shap_values, X_train, interaction_index='HCT')
import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'

shap.summary_plot(shap_values, X_train)
plt.savefig("shap_summary_plot.pdf", format='pdf', bbox_inches='tight')
######################################
import matplotlib.pyplot as plt
import shap
shap.summary_plot(shap_values, X_train)
 (gcf: get current figure)
fig = plt.gcf()

# (gca: get current axis)
ax = plt.gca()

for label in ax.get_xticklabels():
    label.set_fontname('Times New Roman')
    label.set_fontsize(16)
for label in ax.get_yticklabels():
    label.set_fontname('Times New Roman')
    label.set_fontsize(16)
plt.savefig("shap_summary_plot.pdf", format='pdf', bbox_inches='tight')
plt.close(fig)
#####################################
# Calculate interaction impact shap values
shap_interaction_values = explainer.shap_interaction_values(X_train)

# Plot interaction impact graphs
# Here, we select 'SIRI' and 'HCT' as examples of two features"
shap.dependence_plot(
    ("SIRI", "HCT"), 
    shap_interaction_values, 
    X_train,
    display_features=X_train
)


import itertools

# Obtain pairwise combinations of all features
feature_pairs = list(itertools.combinations(col_xcg, 2))

# For each pair of features, plot their interaction effect graphs
fig, axs = plt.subplots(len(feature_pairs)//2, 2, figsize=(15, len(feature_pairs)*5))
for ax, (feature1, feature2) in zip(axs.flatten(), feature_pairs):
    shap.dependence_plot(
        (feature1, feature2),
        shap_interaction_values, 
        X_train,
        ax=ax
    )
    ax.set_title(f'Interaction between {feature1} and {feature2}')

plt.tight_layout()
plt.show()


import itertools
import os
import matplotlib

if not os.path.exists('interaction_plots'):
    os.makedirs('interaction_plots')

matplotlib.rcParams['font.family'] = 'Times New Roman'


feature_pairs = list(itertools.combinations(col_xcg, 2))


for feature1, feature2 in feature_pairs:
    shap.dependence_plot(
        (feature1, feature2),
        shap_interaction_values, 
        X_train
    )
    plt.title(f'Interaction between {feature1} and {feature2}', fontsize=16)

   
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    plt.savefig(f'interaction_plots/{feature1}_{feature2}_interaction.pdf')  # 将图像保存为PDF
    plt.close()

####################
shap.summary_plot(shap_interaction_values, X_train, plot_type="compact_dot")
##################################################
################################
##########################
#Randomly extract 100 samples, 50 positive and 50 negative cases
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
import os

# 切换工作路径
os.chdir(r'/home/changshu/python2022/huanhu_data/hypertension_final_code')

col_xcg = ['SIRI', 'HCT', 'RDW_CV', 'PLT', 'BAS_p', 'IG_p', 'EOS']
X_train = pd.read_csv('X_train.csv')[col_xcg]
y_train = pd.read_csv('y_train.csv')


with open('robust_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open("save/SR1_xgb_xcg_model.pkl", 'rb') as file:  
    xgboost_final_xcg = pickle.load(file)


df_class_0 = X_train[y_train.iloc[:, 0] == 0]
df_class_1 = X_train[y_train.iloc[:, 0] == 1]

df_class_0_sample = resample(df_class_0, replace=False, n_samples=50, random_state=142)
df_class_1_sample = resample(df_class_1, replace=False, n_samples=50, random_state=142)


if not os.path.exists('sensitive_analysis'):
    os.makedirs('sensitive_analysis')


for feature in col_xcg:
    plt.figure(figsize=(10, 6))

    for i in range(df_class_0_sample.shape[0]):
 
        sample = df_class_0_sample.iloc[i].copy().to_frame().T

   
        min_value = X_train[feature].min()
        max_value = X_train[feature].max()

        values = np.linspace(min_value, max_value, 100)

        predictions = []

      
        for value in values:
   
            sample[feature] = value

  
            sample_scaled = pd.DataFrame(scaler.transform(sample), columns=sample.columns)

            predictions.append(xgboost_final_xcg.predict_proba(sample_scaled)[0, 1])

        plt.plot(values, predictions, color='blue', alpha=0.5)


    for i in range(df_class_1_sample.shape[0]):
      
        sample = df_class_1_sample.iloc[i].copy().to_frame().T

     
        min_value = X_train[feature].min()
        max_value = X_train[feature].max()

  
        values = np.linspace(min_value, max_value, 100)

     
        predictions = []

 
        for value in values:
        
            sample[feature] = value

     
            sample_scaled = pd.DataFrame(scaler.transform(sample), columns=sample.columns)

          
            predictions.append(xgboost_final_xcg.predict_proba(sample_scaled)[0, 1])

      
        plt.plot(values, predictions, color='red', alpha=0.5)


    plt.legend(['No AIS', 'AIS'])
    plt.xlabel(feature)
    plt.ylabel('Prediction')
    plt.title(f'Impact of {feature} on Prediction')
    
  
    plt.savefig(f'sensitive_analysis/{feature}_impact.pdf', dpi=300)

    plt.close()




#Global sensitivity analysis



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.metrics import balanced_accuracy_score

os.chdir(r'/home/changshu/python2022/huanhu_data/hypertension_final_code')

# Define features
features = ['SIRI', 'HCT', 'RDW_CV', 'PLT', 'BAS_p', 'IG_p', 'EOS']

# Load data
X_train = pd.read_csv('X_train.csv')[features]
y_train = pd.read_csv('y_train.csv')

# Load the scaler and transform X_train
with open('robust_scaler.pkl', 'rb') as file:
    scaler_loaded = pickle.load(file)

X_train_scaled = pd.DataFrame(scaler_loaded.transform(X_train), columns=features)

# Load the model
Pkl_Filename = "save/SR1_xgb_xcg_model.pkl"  
with open(Pkl_Filename, 'rb') as file:  
    xgboost_final_xcg = pickle.load(file)

# Apply SMOTE on the scaled data
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

# Evaluate the initial performance
base_performance = balanced_accuracy_score(y_train_res, xgboost_final_xcg.predict(X_train_res))

# Initialize a dictionary to store feature importances
feature_importances_balanced = {}

# Calculate feature importances
for feature in features:
    # Copy the resampled data
    X_temp = X_train_res.copy()

    # Shuffle feature values
    X_temp[feature] = np.random.permutation(X_train_res[feature].values)

    # Calculate performance after shuffling
    shuffled_performance = balanced_accuracy_score(y_train_res, xgboost_final_xcg.predict(X_temp))

    # The importance of a feature is the decrease in performance
    feature_importances_balanced[feature] = base_performance - shuffled_performance

# Plot feature importances
plt.figure(figsize=(10, 6))
bars = plt.bar(feature_importances_balanced.keys(), feature_importances_balanced.values())

# Add numbers on the top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 3), fontsize=18, va='bottom')

plt.title('Feature importances (Balanced)', fontsize=18)
plt.xlabel('Features', fontsize=20)
plt.ylabel('Decrease in Balanced Accuracy', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
#########################
# Sort the feature importances
sorted_importances = dict(sorted(feature_importances_balanced.items(), key=lambda item: item[1], reverse=True))

# Plot feature importances
plt.figure(figsize=(10, 6))
bars = plt.bar(sorted_importances.keys(), sorted_importances.values())

# Add numbers on the top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 3), fontsize=18, va='bottom')

plt.title('Feature importances (Balanced)', fontsize=18)
plt.xlabel('Features', fontsize=20)
plt.ylabel('Decrease in Balanced Accuracy', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
# Save as pdf
plt.savefig('sensitive_balanced_feature_importances.pdf', dpi=300, format='pdf')

plt.show()

##############
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.metrics import balanced_accuracy_score

os.chdir(r'/home/changshu/python2022/huanhu_data/hypertension_final_code')

# Define features
features = ['SIRI', 'HCT', 'RDW_CV', 'PLT', 'BAS_p', 'IG_p', 'EOS']

# Load data
X_train = pd.read_csv('X_train.csv')[features]
y_train = pd.read_csv('y_train.csv')

# Load the scaler and transform X_train
with open('robust_scaler.pkl', 'rb') as file:
    scaler_loaded = pickle.load(file)

X_train_scaled = pd.DataFrame(scaler_loaded.transform(X_train), columns=features)

# Load the model
Pkl_Filename = "save/SR1_xgb_xcg_model.pkl"  
with open(Pkl_Filename, 'rb') as file:  
    xgboost_final_xcg = pickle.load(file)

# Evaluate the initial performance
base_performance = balanced_accuracy_score(y_train, xgboost_final_xcg.predict(X_train_scaled))

# Initialize a dictionary to store feature importances
feature_importances_balanced = {}

# Calculate feature importances
for feature in features:
    # Copy the resampled data
    X_temp = X_train_scaled.copy()

    # Shuffle feature values
    X_temp[feature] = np.random.permutation(X_train_scaled[feature].values)

    # Calculate performance after shuffling
    shuffled_performance = balanced_accuracy_score(y_train, xgboost_final_xcg.predict(X_temp))

    # The importance of a feature is the decrease in performance
    feature_importances_balanced[feature] = base_performance - shuffled_performance

# Plot feature importances
plt.figure(figsize=(10, 6))
bars = plt.bar(feature_importances_balanced.keys(), feature_importances_balanced.values())

# Add numbers on the top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 3), fontsize=18, va='bottom')

plt.title('Feature importances (Balanced)', fontsize=18)
plt.xlabel('Features', fontsize=20)
plt.ylabel('Decrease in Balanced Accuracy', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
#Sort the feature importances
sorted_importances = dict(sorted(feature_importances_balanced.items(), key=lambda item: item[1], reverse=True))

# Plot feature importances
plt.figure(figsize=(10, 6))
bars = plt.bar(sorted_importances.keys(), sorted_importances.values())

# Add numbers on the top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 3), fontsize=18, va='bottom')

plt.title('Feature importances (Balanced)', fontsize=18)
plt.xlabel('Features', fontsize=20)
plt.ylabel('Decrease in Balanced Accuracy', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
# Save as pdf
plt.savefig('0sensitive_balanced_feature_importances.pdf', dpi=300, format='pdf')

plt.show()
