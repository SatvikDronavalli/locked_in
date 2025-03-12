import numpy as np
import pandas as pd
from IMU_Data import X_train,y_train
import scipy.stats as stats
print(type(X_train),type(y_train))
X_train["outputs"] = y_train
# Example dataset: Features (step_frequency, stride_length) and binary labels (0=normal, 1=impaired)
data = X_train
# Separate features and labels
features = list(X_train.columns)
features.remove("outputs")

# Perform Mann-Whitney U test for each feature
alpha = 0.05  # Significance threshold
selected_features = []

for feature in features:
    class_0 = data[data['outputs'] == 0][feature]  # Feature values for class 0
    class_1 = data[data['outputs'] == 1][feature]  # Feature values for class 1
    
    u_stat, p_value = stats.mannwhitneyu(class_0, class_1, alternative='two-sided')
    
    print(f"Feature: {feature} | U-Statistic: {u_stat} | P-Value: {p_value}")
    
    if p_value < alpha:
        selected_features.append(feature)  # Keep significant features

print("\nSelected Features for Classification:", selected_features)
