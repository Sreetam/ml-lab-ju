import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from time import time
import matplotlib.pyplot as plt

# Load the diabetes dataset
file_path = './datasets/diabetes_dataset.csv'
data = pd.read_csv(file_path)

# Separate features and target
X = data.iloc[:, :-1].values  # Assuming the last column is the target
y = data.iloc[:, -1].values
y = np.where(y == 0, -1, 1)  # Convert to -1, 1 for AdaBoost

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KFold cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

class CustomAdaBoost:
    def __init__(self, base_estimator, n_estimators):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators

    def fit(self, X, y):
        n_samples = X.shape[0]
        self.models = []
        self.model_weights = []

        # Initialize weights
        w = np.ones(n_samples) / n_samples

        for k in range(self.n_estimators):
            # Train the base estimator with sample weights
            self.base_estimator.fit(X, y, sample_weight=w)
            predictions = self.base_estimator.predict(X)

            # Compute weighted error
            incorrect = (predictions != y).astype(int)
            error = np.dot(w, incorrect) / np.sum(w)

            # Stop if error is too high
            if error > 0.5:
                break

            # Calculate model weight
            model_weight = 0.5 * np.log((1 - error) / max(error, 1e-10))

            # Update sample weights
            w = w * np.exp(-model_weight * y * predictions)
            w /= np.sum(w)  # Normalize weights

            # Save the model and its weight
            self.models.append(self.base_estimator)
            self.model_weights.append(model_weight)

    def predict(self, X):
        # Combine predictions using weighted majority vote
        final_predictions = np.zeros(X.shape[0])
        for model, weight in zip(self.models, self.model_weights):
            final_predictions += weight * model.predict(X)
        return np.sign(final_predictions)

    def get_params(self, deep=True):
        return {"base_estimator": self.base_estimator, "n_estimators": self.n_estimators}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

class CustomLogisticRegression:
    def __init__(self, lr=0.01, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, X, y, sample_weight=None):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        if sample_weight is None:
            sample_weight = np.ones(n_samples)

        for _ in range(self.n_iter):
            linear_model = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(linear_model)

            # Gradient descent with weights
            errors = y - predictions
            dw = -np.dot(X.T, errors * sample_weight) / n_samples
            db = -np.sum(errors * sample_weight) / n_samples

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        predictions = self._sigmoid(linear_model)
        return np.where(predictions >= 0.5, 1, -1)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

# Function to evaluate models
def evaluate_model(model, X_train, y_train, X_test, y_test, kf, model_name):
    start_train = time()
    model.fit(X_train, y_train)
    end_train = time()

    start_test = time()
    predictions = model.predict(X_test)
    end_test = time()

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')

    return accuracy, np.mean(cv_scores), f1, end_train - start_train, end_test - start_test

# Models to evaluate
results = {}

# AdaBoost with Decision Trees
base_estimator_tree = DecisionTreeClassifier(max_depth=1)
adaboost_tree = CustomAdaBoost(base_estimator=base_estimator_tree, n_estimators=50)
results['AdaBoost + Decision Tree'] = evaluate_model(adaboost_tree, X_train, y_train, X_test, y_test, kf, 'AdaBoost + Decision Tree')

# AdaBoost with Logistic Regression
base_estimator_logistic = CustomLogisticRegression(lr=0.1, n_iter=1000)
adaboost_logistic = CustomAdaBoost(base_estimator=base_estimator_logistic, n_estimators=50)
results['AdaBoost + Logistic Regression'] = evaluate_model(adaboost_logistic, X_train, y_train, X_test, y_test, kf, 'AdaBoost + Logistic Regression')

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
results['Random Forest'] = evaluate_model(rf, X_train, y_train, X_test, y_test, kf, 'Random Forest')

# Display Results
for model_name, (test_acc, cv_acc, f1, train_time, test_time) in results.items():
    print(f"{model_name}:")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Cross-Validation Accuracy: {cv_acc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Training Time: {train_time:.4f} seconds")
    print(f"  Inference Time: {test_time:.4f} seconds")
    print()

# Plotting Results
models = list(results.keys())
test_accuracies = [results[m][0] for m in models]
cv_accuracies = [results[m][1] for m in models]
f1_scores = [results[m][2] for m in models]

plt.figure(figsize=(10, 6))

plt.bar(models, test_accuracies, color='blue', alpha=0.6, label='Test Accuracy')
plt.bar(models, f1_scores, color='red', alpha=0.6, label='F1 Score', width=0.4)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Score')
plt.title('Model Comparison')
plt.legend()
plt.tight_layout()
plt.show()