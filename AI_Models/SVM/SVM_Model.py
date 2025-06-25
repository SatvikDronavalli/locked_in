import numpy as np
import tensorflow as tf
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Make predictions
from sklearn.model_selection import GridSearchCV
from IMU_Data.IMU_data import X_train, X_test, y_train, y_test, params
print(len(X_train),len(X_test),len(y_train),len(y_test))
#sns.pairplot(X_train,hue='at risk of falls')
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
svm_model = SVC(kernel="rbf", C=1.0, gamma="scale")  # Use RBF kernel for non-linearity
svm_model.fit(X_train, y_train)
# Predictions
y_pred = svm_model.predict(X_test)
param_grid = {
    'C': [0.1, 1, 10, 100], 
    'gamma': ['scale', 0.01, 0.1, 1],
    'kernel': ['rbf']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)
svm_model = grid_search.best_estimator_

# Make predictions
y_pred = svm_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Precision, Recall, F1-score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# Full classification report
# meta learning
'''
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
print("\nClassification Report:\n", classification_report(y_test, y_pred))'''


#TODO: Determine weights that are most important

perm_importance = permutation_importance(svm_model, X_test, y_test)
sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(params[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")
plt.show()
