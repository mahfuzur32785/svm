import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from tqdm import tqdm

import pennylane as qml

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
scaler = StandardScaler()
df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])

# Split data
X = df.iloc[:, :-1]
y = df['Outcome'].apply(lambda x: 1 if x == 1 else -1)  # Required for QSVR kernel compatibility
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)

# Set up PennyLane quantum kernel
n_qubits = X_train.shape[1]
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def kernel_circuit(x1, x2):
    qml.templates.AngleEmbedding(x1, wires=range(n_qubits))
    qml.adjoint(qml.templates.AngleEmbedding)(x2, wires=range(n_qubits))
    return qml.probs(wires=range(n_qubits))

# --- Visualize the quantum circuit (ASCII + matplotlib) ---
print("\nQuantum Circuit Diagram (text):")
example_x1 = X_train[0]
example_x2 = X_train[1]
print(qml.draw(kernel_circuit)(example_x1, example_x2))

print("\nRendering quantum circuit diagram...")
qml.draw_mpl(kernel_circuit)(example_x1, example_x2)
plt.title("Quantum Kernel Circuit")
plt.tight_layout()
plt.show()

# --- Define quantum kernel matrix function ---
def quantum_kernel_matrix(X1, X2):
    kernel_matrix = []
    for i, x1 in enumerate(tqdm(X1, desc="Computing Kernel Rows")):
        row = []
        for x2 in tqdm(X2, desc=f"Row {i+1}/{len(X1)}", leave=False):
            row.append(kernel_circuit(x1, x2)[0])
        kernel_matrix.append(row)
    return np.array(kernel_matrix)

# Compute kernel matrices
print("Computing training kernel matrix...")
K_train = quantum_kernel_matrix(X_train, X_train)
print("Computing test kernel matrix...")
K_test = quantum_kernel_matrix(X_test, X_train)

# Train QSVR model using custom kernel
qsvm = SVC(kernel="precomputed")
qsvm.fit(K_train, y_train)

# Predict and evaluate
y_pred = qsvm.predict(K_test)
print(f"\nQSVR Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=["No Diabetes", "Diabetes"],
            yticklabels=["No Diabetes", "Diabetes"])
plt.title("Confusion Matrix - QSVR")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
