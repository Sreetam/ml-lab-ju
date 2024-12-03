import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime

# Load dataset
file_path = os.path.join(os.path.dirname(__file__), 'datasets', 'diabetes_dataset.csv')
data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
X, y = data[:, :-1], data[:, -1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Neural Network Class
class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, nodes_per_layer, output_size, activation, momentum=0.9, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.activation = activation
        self.weights = []
        self.biases = []
        self.velocities_w = []
        self.velocities_b = []

        # Initialize weights and biases
        node_size = [input_size] + [nodes_per_layer] * hidden_layers + [output_size]
        for i in range(len(node_size) - 1):
            self.weights.append(np.random.randn(node_size[i], node_size[i+1]) * 0.1)
            self.biases.append(np.zeros((1, node_size[i+1])))
            self.velocities_w.append(np.zeros((node_size[i], node_size[i+1])))
            self.velocities_b.append(np.zeros((1, node_size[i+1])))

    def forward(self, X):
        self.z = []  # Linear combinations
        self.a = [X]  # Activations

        for i in range(len(self.weights)):
            z = np.dot(self.a[-1], self.weights[i]) + self.biases[i]
            self.z.append(z)

            if i < len(self.weights) - 1:  # Hidden layers
                a = relu(z) if self.activation == 'relu' else sigmoid(z)
            else:  # Output layer
                a = sigmoid(z)
            self.a.append(a)

        return self.a[-1]

    def backward(self, y_true):
        m = y_true.shape[0]
        deltas = [self.a[-1] - y_true.reshape(-1, 1)]  # Output layer error

        # Backpropagate errors
        for i in reversed(range(len(self.weights) - 1)):
            if self.activation == 'relu':
                delta = np.dot(deltas[-1], self.weights[i+1].T) * relu_derivative(self.z[i])
            else:
                delta = np.dot(deltas[-1], self.weights[i+1].T) * sigmoid_derivative(self.a[i+1])
            deltas.append(delta)
        deltas.reverse()

        # Update weights and biases with momentum
        for i in range(len(self.weights)):
            dw = np.dot(self.a[i].T, deltas[i]) / m
            db = np.sum(deltas[i], axis=0, keepdims=True) / m

            self.velocities_w[i] = self.momentum * self.velocities_w[i] - self.learning_rate * dw
            self.velocities_b[i] = self.momentum * self.velocities_b[i] - self.learning_rate * db

            self.weights[i] += self.velocities_w[i]
            self.biases[i] += self.velocities_b[i]

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / m

    def train(self, X, y, epochs=50):
        losses = []
        for epoch in range(epochs):
            y_pred = self.forward(X)
            self.backward(y)
            loss = self.compute_loss(y, y_pred)
            losses.append(loss)
            if epoch % 10 == 0:
                pass
                # print(f'Epoch {epoch}, Loss: {loss}')
        return losses

# Training and evaluation
input_size = X_train.shape[1]
output_size = 1

# Case 1: All sigmoid
print("Training model with all sigmoid activations...")
nn_sigmoid = NeuralNetwork(input_size, hidden_layers=2, nodes_per_layer=16, output_size=output_size, activation='sigmoid', learning_rate=0.001)
losses_sigmoid = nn_sigmoid.train(X_train, y_train, epochs=1000)

# Case 2: ReLU hidden layers, sigmoid output
print("Training model with ReLU hidden layers and sigmoid output...")
nn_relu = NeuralNetwork(input_size, hidden_layers=2, nodes_per_layer=16, output_size=output_size, activation='relu', learning_rate=0.001)
losses_relu = nn_relu.train(X_train, y_train, epochs=1000)

# Plotting epoch vs loss
plt.figure(figsize=(12, 6))
plt.plot(losses_sigmoid, label='Sigmoid Loss')
plt.plot(losses_relu, label='ReLU + Sigmoid Loss')
plt.title('Epoch vs Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.savefig(f"outputs/{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
plt.show()
