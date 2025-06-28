# diabetes_classifier.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes

# Load diabetes regression dataset
data, target = load_diabetes(return_X_y=True)
feature_names = load_diabetes().feature_names

# Construct DataFrame
df = pd.DataFrame(data, columns=feature_names)
df['Outcome'] = (target > target.mean()).astype(int)  # Convert to binary classification

# Print dataset info
print(f"Dataset shape: {df.shape}")
print("Feature columns:", list(df.columns))

# Check for null values
print("Null values in dataset:", df.isnull().sum().sum())

# Correlation matrix heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# Bar graph for class distribution
df['Outcome'].value_counts().plot(kind='bar', color=['blue', 'orange'])
plt.xlabel("Diabetes (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.title("Diabetes Class Distribution")
plt.tight_layout()
plt.show()

# Normalize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])

# Split data
from sklearn.model_selection import train_test_split
X = df.iloc[:, :-1]
y = df['Outcome'].apply(lambda x: 1 if x == 1 else -1)  # Convert 0 -> -1 (required by quantum SVM kernel)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifiers
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

models = {
    "SVM": SVC()
}

# Accuracy evaluation
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name} Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Detailed classification report
for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"\nPerformance of {name}:\n{classification_report(y_test, y_pred)}")

# Confusion matrices
for name, model in models.items():
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=["No Diabetes", "Diabetes"],
                yticklabels=["No Diabetes", "Diabetes"])
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
