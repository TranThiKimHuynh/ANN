import numpy as np
# Import fashion_mnist
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def load_data():
    # Load Fashion MNIST directly
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    
    # Normalize the data
    X_train = X_train.reshape(-1, 28 * 28).astype(np.float32) / 255.0
    X_test = X_test.reshape(-1, 28 * 28).astype(np.float32) / 255.0
    return X_train, y_train, X_test, y_test

# One-hot encoding for labels
def one_hot_encode(y, num_classes):
    encoder = OneHotEncoder(sparse_output=False, categories='auto')
    y = y.reshape(-1, 1)
    return encoder.fit_transform(y)

# Accuracy calculation
def calculate_accuracy(predictions, true_labels):
    return np.mean(predictions == true_labels) * 100

# Split data into train and validation
def split_data(X, y, validation_size=0.2):
    return train_test_split(X, y, test_size=validation_size, random_state=42)

# Activation Functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Loss Function
def cross_entropy_loss(predictions, targets):
    n_samples = targets.shape[0]
    log_likelihood = -np.log(predictions[range(n_samples), targets.argmax(axis=1)])
    return np.sum(log_likelihood) / n_samples

# Gradient of Loss Function
def gradient_cross_entropy(predictions, targets):
    return predictions - targets

# Initialize weights and biases using He Initialization
def initialize_weights(input_size, hidden1_size, hidden2_size, output_size):
    weights = {
        'W1': np.random.randn(input_size, hidden1_size) * np.sqrt(2. / input_size),
        'b1': np.zeros((1, hidden1_size)),
        'W2': np.random.randn(hidden1_size, hidden2_size) * np.sqrt(2. / hidden1_size),
        'b2': np.zeros((1, hidden2_size)),
        'W3': np.random.randn(hidden2_size, output_size) * np.sqrt(2. / hidden2_size),
        'b3': np.zeros((1, output_size))
    }
    return weights


# Forward Propagation
def forward_propagation(X, weights):
    Z1 = np.dot(X, weights['W1']) + weights['b1']
    A1 = relu(Z1)
    Z2 = np.dot(A1, weights['W2']) + weights['b2']
    A2 = relu(Z2)
    Z3 = np.dot(A2, weights['W3']) + weights['b3']
    A3 = softmax(Z3)
    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2, 'Z3': Z3, 'A3': A3}
    return A3, cache

# Backward Propagation
def backward_propagation(X, Y, cache, weights,loss_entropy):
    m = X.shape[0]
    gradients = {}
    
    dZ3 = gradient_cross_entropy(cache['A3'], Y)* loss_entropy
    gradients['dW3'] = np.dot(cache['A2'].T, dZ3) / m
    gradients['db3'] = np.sum(dZ3, axis=0, keepdims=True) / m

    
    dA2 = np.dot(dZ3, weights['W3'].T)
    dZ2 = dA2 * relu_derivative(cache['Z2'])
    gradients['dW2'] = np.dot(cache['A1'].T, dZ2) / m
    gradients['db2'] = np.sum(dZ2, axis=0, keepdims=True) / m
    
    dA1 = np.dot(dZ2, weights['W2'].T)
    dZ1 = dA1 * relu_derivative(cache['Z1'])
    gradients['dW1'] = np.dot(X.T, dZ1) / m
    gradients['db1'] = np.sum(dZ1, axis=0, keepdims=True) / m
    
    return gradients

# Update Weights
def update_weights(weights, gradients, learning_rate):
    for key in weights.keys():
        weights[key] -= learning_rate * gradients['d' + key]
    return weights

# Training Loop (with accuracy)
def train(X_train, y_train, X_val, y_val, epochs, learning_rate, input_size, hidden1_size, hidden2_size, output_size):
    weights = initialize_weights(input_size, hidden1_size, hidden2_size, output_size)
    
    for epoch in range(epochs):
        # Forward propagation
        predictions, cache = forward_propagation(X_train, weights)
        
        # Compute training loss
        train_loss = cross_entropy_loss(predictions, y_train)
        train_accuracy = calculate_accuracy(np.argmax(predictions, axis=1), np.argmax(y_train, axis=1))
        
        # Backward propagation
        gradients = backward_propagation(X_train, y_train, cache, weights,train_loss)
        
        # Update weights
        weights = update_weights(weights, gradients, learning_rate)
        
        # Validation
        val_predictions, _ = forward_propagation(X_val, weights)
        val_loss = cross_entropy_loss(val_predictions, y_val)
        val_accuracy = calculate_accuracy(np.argmax(val_predictions, axis=1), np.argmax(y_val, axis=1))
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
    
    return weights


# Predict function
def predict(X, weights):
    predictions, _ = forward_propagation(X, weights)
    return np.argmax(predictions, axis=1)

# Accuracy calculation
def calculate_accuracy(predictions, true_labels):
    return np.mean(predictions == true_labels) * 100
# Main Function
if __name__ == "__main__":
    # Load data
    X_train, y_train, X_test, y_test = load_data()
    
    # One-hot encode the labels
    y_train_one_hot = one_hot_encode(y_train, num_classes=10)
    y_test_one_hot = one_hot_encode(y_test, num_classes=10)
    
    print("Training data shape:", X_train.shape, y_train_one_hot.shape)
    print("Test data shape:", X_test.shape, y_test_one_hot.shape)
    
    # Split into train and validation sets
    X_train_split, X_val, y_train_split, y_val = split_data(X_train, y_train_one_hot)
    
    print("Train split shape:", X_train_split.shape, y_train_split.shape)
    print("Validation split shape:", X_val.shape, y_val.shape)
    
    # Hyperparameters
    input_size = 28 * 28
    hidden1_size = 128
    hidden2_size = 128
    output_size = 10
    epochs = 50
    learning_rate = 0.1
    
    # Train model
    trained_weights = train(X_train_split, y_train_split, X_val, y_val, epochs, learning_rate, input_size, hidden1_size, hidden2_size, output_size)
    
    # Test model accuracy
    test_predictions = predict(X_test, trained_weights)
    test_accuracy = calculate_accuracy(test_predictions, y_test)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
