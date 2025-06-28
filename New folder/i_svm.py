# svm.py

import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set random seed for reproducibility
np.random.seed(42)

# Load the Iris dataset
X, y = load_iris(return_X_y=True)

# Use only the first two classes (first 100 samples)
X = X[:100, :]
y = y[:100]

# Scale input features
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# Scale labels to -1 and 1
y_scaled = 2 * (y - 0.5)

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.25, random_state=42
)

# Train classical SVM (you can change kernel to 'linear', 'poly', or 'sigmoid' if needed)
clf = SVC(kernel="linear", C=1.0)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Print results
print("Predictions:", y_pred)
print("Actual     :", y_test)
print("Accuracy   :", acc)
