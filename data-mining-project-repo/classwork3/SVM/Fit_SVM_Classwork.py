#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("student_data.csv")

# Prepare the data
X = df[["Hours_Studied", "Review_Session"]]  # Features
y = df["Results"]  # Target variable

# Fit SVM with Linear Kernel
svm_linear = SVC(kernel="linear")
svm_linear.fit(X, y)

# Predict and evaluate
y_pred_linear = svm_linear.predict(X)
accuracy_linear = accuracy_score(y, y_pred_linear)
print("Linear SVM Accuracy:", accuracy_linear)

# Fit SVM with RBF Kernel and Grid Search for best gamma
param_grid = {"gamma": [0.1, 1, 10, 100]}  # Parameter grid for gamma
svm_rbf = SVC(kernel="rbf")

# Perform grid search with 5-fold cross-validation ##K-fold
grid_search = GridSearchCV(svm_rbf, param_grid, cv=5)
grid_search.fit(X, y)

# Best gamma and accuracy
best_gamma = grid_search.best_params_["gamma"]
print("Best Gamma for RBF Kernel:", best_gamma)

# Predict and evaluate with the best gamma
y_pred_rbf = grid_search.predict(X)
accuracy_rbf = accuracy_score(y, y_pred_rbf)
print("RBF SVM Accuracy (with best gamma):", accuracy_rbf)

# Cross-validation scores for RBF SVM with best gamma
cv_scores = cross_val_score(grid_search.best_estimator_, X, y, cv=5)
print("Cross-Validation Scores (RBF SVM):", cv_scores)
print("Mean Cross-Validation Accuracy (RBF SVM):", cv_scores.mean())


# In[ ]:




