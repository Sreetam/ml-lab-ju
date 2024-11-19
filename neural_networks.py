import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import tensorflow as tf

# Check for GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if len(tf.config.list_physical_devices('GPU')) > 0:
    print("Using GPU for training.")
else:
    print("No GPU found. Training will use CPU.")

# Load dataset
data = pd.read_csv('diabetes.csv')  # Replace with your dataset path

# Separate features and target
X = data.drop('Outcome', axis=1).values
y = data['Outcome'].values

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def build_model(hidden_layers, nodes_per_layer, activation='sigmoid'):
    model = Sequential()
    # Input layer
    model.add(Dense(nodes_per_layer, input_dim=X_train.shape[1], activation=activation))
    # Hidden layers
    for _ in range(hidden_layers - 1):
        model.add(Dense(nodes_per_layer, activation=activation))
    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model with momentum
    optimizer = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Case 1: All nodes are sigmoid
with tf.device('/GPU:0'):  # Explicitly specify GPU usage
    model_sigmoid = build_model(hidden_layers=2, nodes_per_layer=16, activation='sigmoid')
    history_sigmoid = model_sigmoid.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=0)

# Case 2: Hidden nodes are ReLU, output nodes are sigmoid
with tf.device('/GPU:0'):  # Explicitly specify GPU usage
    model_relu_sigmoid = Sequential()
    model_relu_sigmoid.add(Dense(16, input_dim=X_train.shape[1], activation='relu'))
    model_relu_sigmoid.add(Dense(16, activation='relu'))
    model_relu_sigmoid.add(Dense(1, activation='sigmoid'))

    # Compile with momentum
    optimizer = SGD(learning_rate=0.01, momentum=0.9)
    model_relu_sigmoid.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history_relu_sigmoid = model_relu_sigmoid.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=0)

# Plot epoch vs error
plt.figure(figsize=(12, 6))

# Case 1: All sigmoid
plt.plot(history_sigmoid.history['loss'], label='Sigmoid Train Loss')
plt.plot(history_sigmoid.history['val_loss'], label='Sigmoid Validation Loss')

# Case 2: ReLU + Sigmoid
plt.plot(history_relu_sigmoid.history['loss'], label='ReLU + Sigmoid Train Loss')
plt.plot(history_relu_sigmoid.history['val_loss'], label='ReLU + Sigmoid Validation Loss')

# Configure plot
plt.title('Epoch vs Error (Loss)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()
