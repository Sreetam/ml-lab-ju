import numpy as np  
import matplotlib.pyplot as plt
from datetime import datetime
np.random.seed(30)
X_train1 = np.array([
    [0.3, 0.2],
    [0.8, 0.9],
    [0.1, 0.8],
    [0.95, 0.9],
    [0.9, 0.4],
    [0.7, 0.8],
    [0.4, 0.3],
    [0.2, 0.85],
    [0.82, 0.93],
    [0.75, 0.25]
])
y_train1 = np.array([-1, 1, -1, 1, -1, 1, -1, -1, 1, -1])

X_train2 = np.array([
    [0.3, 0.2],
    [0.8, 0.9],
    [0.1, 0.8],
    [0.95, 0.9],
    [0.9, 0.4],
    [0.7, 0.8],
    [0.4, 0.3],
    [0.2, 0.85],
    [0.82, 0.93],
    [0.75, 0.25]
])
y_train2 = np.array([-1, -1, 1, -1, 1, -1, -1, 1, -1, 1])

# Initialize weights and bias
weights_perceptron_train1 = np.random.rand(2)
bias_perceptron_train1 = np.random.rand(1)[0]
weights_perceptron_train2 = np.random.rand(2)
bias_perceptron_train2 = np.random.rand(1)[0]

weights_delta_train1 = np.random.rand(2)
bias_delta_train1 = np.random.rand(1)[0]
weights_delta_train2 = np.random.rand(2)
bias_delta_train2 = np.random.rand(1)[0]

learning_rate = 0.1
epochs = 100

def perceptron_rule_with_bias(weights, bias, x, target, learning_rate):
    prediction = 1 if (np.dot(x, weights) + bias) >= 0 else -1
    error = target - prediction
    weights += learning_rate * error * x
    bias += learning_rate * error
    return weights, bias, error**2

def delta_rule_with_bias(weights, bias, x, target, learning_rate):
    prediction = np.dot(x, weights) + bias
    error = target - prediction
    weights += learning_rate * error * x
    bias += learning_rate * error
    return weights, bias, error**2

def plot_separator(X, y, weights, bias, title, subplot_position):
    plt.subplot(subplot_position)
    for i in range(len(y)):
        if y[i] == -1:
            plt.scatter(X[i, 0], X[i, 1], color='blue', marker='o', label='Class -1' if i == 0 else "")
        else:
            plt.scatter(X[i, 0], X[i, 1], color='red', marker='x', label='Class 1' if i == 0 else "")

    # Separator line
    x_values = np.linspace(0, 1, 100)
    y_values = -(weights[0] * x_values + bias) / weights[1]
    plt.plot(x_values, y_values, color='green', label='Separator')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)
    plt.legend()
    plt.grid(True)

# Training loop
errors_perceptron_train1, errors_perceptron_train2 = [], []
errors_delta_train1, errors_delta_train2 = [], []

for epoch in range(epochs):
    epoch_error_perceptron_train1 = 0
    epoch_error_perceptron_train2 = 0
    epoch_error_delta_train1 = 0
    epoch_error_delta_train2 = 0
    
    for i in range(len(X_train1)):
        # Perceptron training
        weights_perceptron_train1, bias_perceptron_train1, error_p_train1 = perceptron_rule_with_bias(weights_perceptron_train1, bias_perceptron_train1, X_train1[i], y_train1[i], learning_rate)
        weights_perceptron_train2, bias_perceptron_train2, error_p_train2 = perceptron_rule_with_bias(weights_perceptron_train2, bias_perceptron_train2, X_train2[i], y_train2[i], learning_rate)
        
        # Delta training
        weights_delta_train1, bias_delta_train1, error_d_train1 = delta_rule_with_bias(weights_delta_train1, bias_delta_train1, X_train1[i], y_train1[i], learning_rate)
        weights_delta_train2, bias_delta_train2, error_d_train2 = delta_rule_with_bias(weights_delta_train2, bias_delta_train2, X_train2[i], y_train2[i], learning_rate)
        
        # Error accumulation
        epoch_error_perceptron_train1 += error_p_train1
        epoch_error_perceptron_train2 += error_p_train2
        epoch_error_delta_train1 += error_d_train1
        epoch_error_delta_train2 += error_d_train2
    
    errors_perceptron_train1.append(epoch_error_perceptron_train1)
    errors_perceptron_train2.append(epoch_error_perceptron_train2)
    errors_delta_train1.append(epoch_error_delta_train1)
    errors_delta_train2.append(epoch_error_delta_train2)

plt.figure(figsize=(18, 12))

plt.subplot(2, 2, 1)
plt.plot(range(epochs), errors_perceptron_train1, label='Training Set 1 Error (Perceptron)')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Error vs Epochs for Training Set 1 (Perceptron)')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(range(epochs), errors_perceptron_train2, label='Training Set 2 Error (Perceptron)')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Error vs Epochs for Training Set 2 (Perceptron)')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(range(epochs), errors_delta_train1, label='Training Set 1 Error (Delta)')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Error vs Epochs for Training Set 1 (Delta)')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(range(epochs), errors_delta_train2, label='Training Set 2 Error (Delta)')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Error vs Epochs for Training Set 2 (Delta)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(f"outputs/{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
plt.show()

plt.figure(figsize=(18, 12))

plot_separator(X_train1, y_train1, weights_perceptron_train1, bias_perceptron_train1, "Training Set 1 with Separator (Perceptron)", 221)
plot_separator(X_train2, y_train2, weights_perceptron_train2, bias_perceptron_train2, "Training Set 2 with Separator (Perceptron)", 222)

plot_separator(X_train1, y_train1, weights_delta_train1, bias_delta_train1, "Training Set 1 with Separator (Delta)", 223)
plot_separator(X_train2, y_train2, weights_delta_train2, bias_delta_train2, "Training Set 2 with Separator (Delta)", 224)

plt.tight_layout()
plt.savefig(f"outputs/{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
plt.show()

print("Final weights and bias for Training Set 1 (Perceptron):", weights_perceptron_train1, bias_perceptron_train1)
print("Final weights and bias for Training Set 2 (Perceptron):", weights_perceptron_train2, bias_perceptron_train2)
print("Final weights and bias for Training Set 1 (Delta):", weights_delta_train1, bias_delta_train1)
print("Final weights and bias for Training Set 2 (Delta):", weights_delta_train2, bias_delta_train2) 