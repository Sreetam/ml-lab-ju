import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = os.path.join(os.path.dirname(__file__), 'datasets', 'diabetes_dataset.csv')

# Load the dataset
data = pd.read_csv(file_path)

# Preparing the dataset
X = data.drop('Outcome', axis=1).values
y = data['Outcome'].values

# Function to split data with reproducibility
def train_val_test_split(X, y, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    np.random.seed(seed)  # Set seed for reproducibility
    indices = np.random.permutation(len(X))
    train_end = int(train_ratio * len(X))
    val_end = int((train_ratio + val_ratio) * len(X))
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# Split data into train, validation, and test sets with a fixed seed
X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y, seed=42)

# Logistic Regression Model with optional L2 Regularization
class LogisticRegressionWithRegularization:
    def __init__(self, learning_rate=0.01, epochs=1000, regularization_strength=0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization_strength = regularization_strength  # Default no regularization
        self.weights = None
        self.bias = None
        self.train_errors = []
        self.val_errors = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cross_entropy_loss(self, y_true, y_pred, weights):
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)  # avoid log(0)
        base_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        regularization_term = (self.regularization_strength / 2) * np.sum(weights ** 2)
        return base_loss + regularization_term

    def fit(self, X_train, y_train, X_val, y_val):
        n_samples, n_features = X_train.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            # Linear model
            linear_model = np.dot(X_train, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            # Compute gradients with L2 regularization if enabled
            dw = (1 / n_samples) * np.dot(X_train.T, (y_pred - y_train)) + self.regularization_strength * self.weights
            db = (1 / n_samples) * np.sum(y_pred - y_train)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Calculate and store the training and validation error
            train_loss = self.cross_entropy_loss(y_train, y_pred, self.weights)
            val_pred = self.sigmoid(np.dot(X_val, self.weights) + self.bias)
            val_loss = self.cross_entropy_loss(y_val, val_pred, self.weights)
            self.train_errors.append(train_loss)
            self.val_errors.append(val_loss)

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        return [1 if i >= 0.5 else 0 for i in y_pred]

# Initialize and train the model without regularization
model = LogisticRegressionWithRegularization(learning_rate=0.01, epochs=1000, regularization_strength=0)
model.fit(X_train, y_train, X_val, y_val)

# Plotting epoch vs training and validation 
plt.plot(range(model.epochs), model.val_errors, label='Validation Error')
plt.plot(range(model.epochs), model.train_errors, label='Training Error')
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy Loss')
plt.title('Epoch vs Training and Validation Error')
plt.legend()
plt.show()

plt.scatter(range(model.epochs), model.train_errors, label='Training Error', s=10)
plt.scatter(range(model.epochs), model.val_errors, label='Validation Error', s=10)
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy Loss')
plt.title('Epoch vs Training and Validation Error (Scatter Plot)')
plt.legend()
plt.show()

# Predict on the test data and calculate accuracy manually
y_test_pred = model.predict(X_test)
test_accuracy = np.mean(y_test == y_test_pred)

print("Test Accuracy:", test_accuracy)
print("Final Training Error:", model.train_errors[-1])
print("Final Validation Error:", model.val_errors[-1])
