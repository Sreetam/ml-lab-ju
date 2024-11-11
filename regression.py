from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import time
import os

# Load dataset and preprocess
file_path = os.path.join(os.path.dirname(__file__), 'datasets', 'Housing.csv')
data = pd.read_csv(file_path)
categorical_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
df = pd.get_dummies(data, columns=categorical_columns, dtype=int)
normalized_df = (df - df.min()) / (df.max() - df.min())

# Function to split dataset with reproducibility
def train_val_test_split(df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    np.random.seed(seed)
    train = df.sample(frac=train_ratio, random_state=seed)
    rest = df.drop(train.index)
    test = rest.sample(frac=test_ratio / (val_ratio + test_ratio), random_state=seed)
    validation = rest.drop(test.index)
    return train, validation, test

train_df, val_df, test_df = train_val_test_split(normalized_df)

# Prepare data and add intercept
y_train, X_train = train_df.iloc[:, 0].values, train_df.iloc[:, 1:].values
y_val, X_val = val_df.iloc[:, 0].values, val_df.iloc[:, 1:].values
y_test, X_test = test_df.iloc[:, 0].values, test_df.iloc[:, 1:].values
X_train_b, X_val_b, X_test_b = np.hstack([np.ones((X_train.shape[0], 1)), X_train]), np.hstack([np.ones((X_val.shape[0], 1)), X_val]), np.hstack([np.ones((X_test.shape[0], 1)), X_test])

# Define Linear Regression GD with Training and Validation Error Tracking
class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iters=1000, batch_size=None, method='stochastic'):
        self.learning_rate, self.n_iters, self.batch_size, self.method = learning_rate, n_iters, batch_size, method
        self.beta, self.train_errors, self.val_errors = None, [], []

    def fit(self, X_train, y_train, X_val, y_val):
        n_samples, n_features = X_train.shape
        self.beta = np.zeros(n_features)
        for epoch in range(self.n_iters):
            if self.method == 'stochastic':
                for i in range(n_samples):
                    xi = X_train[i:i+1]
                    error = xi.dot(self.beta) - y_train[i]
                    gradient = 2 * xi.T.dot(error)
                    self.beta -= self.learning_rate * gradient
            elif self.method == 'batch-stochastic':
                indices = np.random.choice(n_samples, self.batch_size, replace=False)
                xi = X_train[indices]
                error = xi.dot(self.beta) - y_train[indices]
                gradient = (2 / self.batch_size) * xi.T.dot(error)
                self.beta -= self.learning_rate * gradient
            train_error = np.mean((X_train.dot(self.beta) - y_train) ** 2)
            val_error = np.mean((X_val.dot(self.beta) - y_val) ** 2)
            self.train_errors.append(train_error)
            self.val_errors.append(val_error)

    def predict(self, X): return X.dot(self.beta)

# Train models with different methods and track metrics
methods, colors, results = ['stochastic', 'batch-stochastic', 'normal'], ['tab:orange', 'tab:purple', 'tab:red'], {}

for method in methods:
    if method == 'normal':
        # Closed-form solution
        start_time = time.time()
        beta = np.linalg.inv(X_train_b.T.dot(X_train_b)).dot(X_train_b.T).dot(y_train)
        end_time = time.time()
        
        y_test_pred = X_test_b.dot(beta)
        train_mse = np.mean((X_train_b.dot(beta) - y_train) ** 2)
        test_mse = np.mean((y_test_pred - y_test) ** 2)
        
        results[method] = {
            'model': None,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'time_taken': end_time - start_time,
            'y_test_pred': y_test_pred
        }
    else:
        # Stochastic and Batch-Stochastic Gradient Descent
        batch_size = 1 if method == 'batch-stochastic' else None
        model = LinearRegressionGD(learning_rate=0.01, n_iters=1000, batch_size=batch_size, method=method)
        
        start_time = time.time()
        model.fit(X_train_b, y_train, X_val_b, y_val)
        end_time = time.time()
        
        y_test_pred = model.predict(X_test_b)
        results[method] = {
            'model': model,
            'train_mse': np.mean((model.predict(X_train_b) - y_train) ** 2),
            'test_mse': np.mean((y_test_pred - y_test) ** 2),
            'train_errors': model.train_errors,
            'val_errors': model.val_errors,
            'time_taken': end_time - start_time,
            'y_test_pred': y_test_pred
        }

# Display metrics in a table
print("\n{:<20} {:<10} {:<10} {:<15}".format("Method", "Train MSE", "Test MSE", "Time Taken (s)"))
for method, result in results.items(): 
    print("{:<20} {:<10.4f} {:<10.4f} {:<15.4f}".format(method.capitalize(), result['train_mse'], result['test_mse'], result['time_taken']))

# Plot MSE and Time Comparison
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Comparison Across Methods')

# Plot Test and Train MSE for each method
for idx, (method, result) in enumerate(results.items()):
    axs[0].bar(idx - 0.2, result['train_mse'], width=0.4, label='Train MSE' if idx == 0 else "")
    axs[0].bar(idx + 0.2, result['test_mse'], width=0.4, label='Test MSE' if idx == 0 else "")
axs[0].set_xticks(range(len(results)))
axs[0].set_xticklabels([method.capitalize() for method in results.keys()])
axs[0].set_title('Train and Test MSE')
axs[0].legend()

# Plot Time Taken for each method
times = [result['time_taken'] for result in results.values()]
axs[1].bar(range(len(results)), times, color=colors)
axs[1].set_xticks(range(len(results)))
axs[1].set_xticklabels([method.capitalize() for method in results.keys()])
axs[1].set_title('Time Taken')

# Plot Training and Validation Error Progression for Stochastic and Batch-Stochastic
for idx, method in enumerate(['stochastic', 'batch-stochastic']):
    axs[2].plot(results[method]['train_errors'], label=f'{method.capitalize()} Train Error', color=colors[idx])
    axs[2].plot(results[method]['val_errors'], label=f'{method.capitalize()} Val Error', linestyle='--', color=colors[idx])
axs[2].set_title('Training and Validation Error Progression')
axs[2].set_xlabel('Epochs')
axs[2].set_ylabel('MSE')
axs[2].legend()

plt.tight_layout()
plt.show()

# PCA on test data and 3D Plot of Actual vs. Predicted for each method
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test_b)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('3D Plot of Actual vs Predicted Using PCA')

for idx, (method, result) in enumerate(results.items()):
    y_test_pred = result['y_test_pred']
    ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], y_test, color='black', label="Actual Data" if idx == 0 else "", alpha=0.3)
    ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], y_test_pred, color=colors[idx], label=f'{method.capitalize()} Prediction')

ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_zlabel('Target Variable')
ax.legend()
plt.show()
