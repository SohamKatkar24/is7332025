import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

# Load the dataset
df = pd.read_csv("student_data.csv")

# Visualize the data
sns.scatterplot(x="Hours_Studied", y="Results", hue="Review_Session", data=df)
plt.title("Scatter Plot of Hours Studied vs Results")
plt.show()

# Prepare the data
X = df[["Hours_Studied", "Review_Session"]]  # Features
y = df["Results"]  # Target variable

# Fit the logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Output model coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Predict probabilities on the training data
y_pred_prob = model.predict_proba(X)[:, 1]  # Probability of passing (class 1)

# Predict classes on the training data
y_pred = model.predict(X)

# Calculate accuracy and AUC
accuracy = accuracy_score(y, y_pred)
auc = roc_auc_score(y, y_pred_prob)

print("Accuracy:", accuracy)
print("AUC:", auc)

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y, y_pred_prob)
plt.plot(fpr, tpr, label="ROC Curve")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

plt.savefig(r"Logistic_Regression_ROC_Curve.png")
plt.show()






