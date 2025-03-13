#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("D:\Documents\Sem_3\IS733\Classwork 3\student_data.csv")

# Prepare the data for SVM
X = data[['Hours_Studied', 'Review_Session']]
y = data['Results']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Fit SVM with Linear Kernel
linear_svm = SVC(kernel='linear', probability=True)  # Enable probability for ROC curve
linear_svm.fit(X_train, y_train)

# Predictions and evaluation for Linear Kernel
y_pred_linear = linear_svm.predict(X_test)
y_pred_proba_linear = linear_svm.predict_proba(X_test)[:, 1]  # Probabilities for ROC curve
accuracy_linear = accuracy_score(y_test, y_pred_linear)
print("Linear Kernel SVM Results:")
print(f"Accuracy: {accuracy_linear}")
print(classification_report(y_test, y_pred_linear))

# 2. Fit SVM with RBF Kernel and Grid Search for best gamma
# Define the parameter grid for gamma
param_grid = {'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

# Initialize the SVM with RBF kernel
rbf_svm = SVC(kernel='rbf', probability=True)  # Enable probability for ROC curve

# Perform grid search with 5-fold cross-validation
grid_search = GridSearchCV(rbf_svm, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Best gamma value
best_gamma = grid_search.best_params_['gamma']
print(f"Best gamma value from grid search: {best_gamma}")

# Predictions and evaluation for RBF Kernel with best gamma
y_pred_rbf = grid_search.predict(X_test)
y_pred_proba_rbf = grid_search.predict_proba(X_test)[:, 1]  # Probabilities for ROC curve
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
print("RBF Kernel SVM Results:")
print(f"Accuracy: {accuracy_rbf}")
print(classification_report(y_test, y_pred_rbf))

# ROC Curve for Linear Kernel
fpr_linear, tpr_linear, _ = roc_curve(y_test, y_pred_proba_linear)
roc_auc_linear = auc(fpr_linear, tpr_linear)

# ROC Curve for RBF Kernel
fpr_rbf, tpr_rbf, _ = roc_curve(y_test, y_pred_proba_rbf)
roc_auc_rbf = auc(fpr_rbf, tpr_rbf)

# Plot ROC Curves
plt.figure(figsize=(8, 6))
plt.plot(fpr_linear, tpr_linear, color='blue', lw=2, label=f'Linear Kernel (AUC = {roc_auc_linear:.2f})')
plt.plot(fpr_rbf, tpr_rbf, color='red', lw=2, label=f'RBF Kernel (AUC = {roc_auc_rbf:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

plt.savefig(r"D:\Documents\Sem_3\IS733\Classwork 3\SVM.png")
plt.show()


# In[ ]:




