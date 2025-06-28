# svm.py

import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pennylane as qml

# Set random seed for reproducibility
# np.random.seed(42)

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

# Set number of qubits based on feature size
n_qubits = X_train.shape[1]

# Set up quantum device
dev = qml.device("default.qubit", wires=n_qubits)

# Define the quantum kernel function
@qml.qnode(dev)
def kernel(x1, x2):
    qml.AngleEmbedding(features=x1, wires=range(n_qubits))
    qml.adjoint(qml.AngleEmbedding)(features=x2, wires=range(n_qubits))
    return qml.probs(wires=range(n_qubits))

# Define function to compute kernel matrix
def kernel_matrix(A, B):
    return np.array([[kernel(a, b)[0] for b in B] for a in A])

# Train the quantum SVM
qsvm = SVC(kernel=kernel_matrix)
qsvm.fit(X_train, y_train)

# Predict and evaluate
y_pred = qsvm.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Print results
print("Predictions:", y_pred)
print("Actual     :", y_test)
print("Accuracy   :", acc)
