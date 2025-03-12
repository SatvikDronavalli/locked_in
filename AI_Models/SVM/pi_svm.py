from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import pandas as pd
from sklearn.pipeline import Pipeline
# Make predictions
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from IMU_Data import new_X_train,new_X_test,new_y_train,new_y_test
scaler = StandardScaler()
X_train = scaler.fit_transform(new_X_train)
X_test = scaler.transform(new_X_test)
pipeline = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel='rbf',probability=True))])
# Predictions
param_grid = {
    'svc__C': [0.1, 1, 10, 100], 
    'svc__gamma': ['scale', 0.01, 0.1, 1],
    'svc__kernel': ['rbf']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, new_y_train)
svm_model = grid_search.best_estimator_

# Make predictions
y_pred = svm_model.predict(X_test)
# Accuracy
accuracy = accuracy_score(new_y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Precision, Recall, F1-score
precision = precision_score(new_y_test, y_pred)
recall = recall_score(new_y_test, y_pred)
f1 = f1_score(new_y_test, y_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# testing on physical data

temp_X = scaler.fit_transform(pd.read_csv("Shaurya's GaitGuard Testing Data - Sheet1 (1).csv"))
probabilities = svm_model.predict_proba(temp_X)
for i, prob in enumerate(probabilities):
    print(f"Sample {i+1}: Class 0 Probability = {prob[0]:.4f}, Class 1 Probability = {prob[1]:.4f}")
print(svm_model.predict(temp_X))
