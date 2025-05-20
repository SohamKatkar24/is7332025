#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns

# 1) Load and clean
df = pd.read_csv(r"Housing.csv")
df = df.dropna()
print("Data shape:", df.shape)

# 2) Scale target to millions
df['price'] = df['price'] / 1e6
print("Median Price (M$):", df['price'].median())

# 3) EDA
plt.figure(figsize=(8, 5))
sns.histplot(df['price'], kde=True)
plt.title('Distribution of Property Prices (M$)')
plt.xlabel('Price (M$)')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

# 4) Features & target
X = df.drop(columns=["price"])
y = df["price"]

# 5) Detect feature types
numeric_feats     = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_feats = X.select_dtypes(include=['object','category']).columns.tolist()
print("Numeric features:", numeric_feats)
print("Categorical features:", categorical_feats)

# 6) Preprocessing pipelines
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
preprocessor = ColumnTransformer([
    ("num", num_pipe, numeric_feats),
    ("cat", cat_pipe, categorical_feats)
])

# 7) Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8) Models & parameter grids
models = {
    'LinearRegression': LinearRegression(),
    'Ridge'           : Ridge(),
    'Lasso'           : Lasso(),
    'DecisionTree'    : DecisionTreeRegressor(random_state=42),
    'RandomForest'    : RandomForestRegressor(random_state=42),
    'GradientBoosting': GradientBoostingRegressor(random_state=42)
}

param_grids = {
    'LinearRegression': {},
    'Ridge'           : {'model__alpha': [0.1, 1.0, 10.0]},
    'Lasso'           : {'model__alpha': [0.01, 0.1, 1.0]},
    'DecisionTree'    : {'model__max_depth': [None, 5, 10]},
    'RandomForest'    : {
        'model__n_estimators': [50, 100],
        'model__max_depth'  : [None, 5, 10]
    },
    'GradientBoosting': {
        'model__n_estimators' : [100, 200],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth'    : [3, 5, 7]
    }
}

# 9) Fit, tune, evaluate
results = {}
for name, estimator in models.items():
    pipe = Pipeline([
        ("preproc", preprocessor),
        ("model",   estimator)
    ])
    grid = GridSearchCV(
        pipe,
        param_grid=param_grids[name],
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )
    print(f"Tuning {name}…")
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    
    results[name] = {
        "MAE (M$)"    : mae,
        "MSE (M$²)"   : mse,
        "R²"          : r2,
        "Best Params" : grid.best_params_
    }

# 10) Build results DataFrame
results_df = pd.DataFrame(results).T
results_df["R²"] = results_df["R²"].astype(float)

print("\nModel comparison:")
print(results_df)

# 11) Plot MAE and R²
fig, axes = plt.subplots(1, 2, figsize=(14,5))
results_df["MAE (M$)"].plot(kind="bar", ax=axes[0], title="MAE by Model")
axes[0].set_ylabel("MAE (M$)")
results_df["R²"].plot(kind="bar", ax=axes[1], title="R² by Model", color="C2")
axes[1].set_ylabel("R²")
plt.tight_layout()
plt.show()

# 12) Identify best model by R²
best = results_df["R²"].idxmax()
print(f"\n★ Best model by R²: {best} (R² = {results_df.loc[best,'R²']:.3f})")


# In[ ]:




