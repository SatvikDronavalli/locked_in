import numpy as np
import tensorflow as tf
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Make predictions
from sklearn.model_selection import GridSearchCV
from Force_Data import gait_x_train, gait_y_train, gait_x_test, gait_y_test
from IMU_Data import X_train, X_test, y_train, y_test
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

# Full classification report
# meta learning
print("\nClassification Report:\n", classification_report(y_test, y_pred))
