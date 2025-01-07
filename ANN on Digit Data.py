
# reading the dataset
import pandas as pd

df = pd.read_csv("digit_data.csv")
df.head()

# convert the dataframe into numpy array

import numpy as np

data = np.array(df)
data

data.shape

var1 = data.shape[0]
var1

var2 = data.shape[1]
var2

# import numpy as np

# arr = np.arange(9).reshape((3, 3))
# arr

# np.random.shuffle(arr)
# arr

# shuffling the data
import numpy as np

np.random.shuffle(data)
data

test_set = data[0:1000]
test_set.shape

train_set = data[1000:]
train_set.shape

# transpose the train and test set

test_trans = np.transpose(test_set)
train_trans = np.transpose(train_set)

print(test_trans.shape)
print(train_trans.shape)

test_trans

#split thr train set in X an Y label

Y_train = train_trans[0]
Y_train.shape

X_train = train_trans[1:]
X_train.shape

Y_test = test_trans[0]
Y_test.shape

X_test = test_trans[1:]
X_test.shape

label_test = test_trans[0]
label_test.shape

feat_test = test_trans[1:]
 feat_test.shape

import numpy as np
import matplotlib.pyplot as plt


pixel_value_counts = {i: 0 for i in range(256)}


for i in range(feat_test.shape[1]):
    unique_pixel_values = np.unique(feat_test[:, i])
    for pixel_value in unique_pixel_values:
        pixel_value_counts[pixel_value] += 1

selected_pixel_values = list(range(0, 256, 10))
selected_counts = [pixel_value_counts[pv] for pv in selected_pixel_values]



plt.figure(figsize=(10, 6))
plt.bar(selected_pixel_values, selected_counts)

plt.xticks(selected_pixel_values)

plt.show()

import random
import matplotlib.pyplot as plt

x = random.randrange(41000)

# Extract the image data (excluding the first element which is probably the label)
image_data = train_set[x, 1:]

# plot the image at this index
plt.imshow(image_data.reshape(28, 28))
plt.show()

from scipy.stats import chi2

def initialize_weights():
    # Degrees of freedom for Chi-squared distribution
    df = 2

    # Initialize weights and biases using Chi-squared distribution
    w1 = np.clip(chi2.rvs(df, size=(10, 784)), 0, 1)
    b1 = np.clip(chi2.rvs(df, size=(10, 1)), 0,1)
    w2 = np.clip(chi2.rvs(df, size=(10, 10)), 0, 1)
    b2 = np.clip(chi2.rvs(df, size=(10, 1)), 0, 1)

    return (w1, b1, w2, b2)

w1, b1, w2, b2 = initialize_weights() # unpack the returned tuple into individual variables
print("W1 shape", w1.shape) # now you can access w1 and check its shape
print("B1 shape", b1.shape)
print("W2 shape", w2.shape)
print("B2 shape", b2.shape)

w2.min()

w2.max()

# histogram of w2

w2_flat = w2.flatten()

plt.hist(w2_flat, bins=10, edgecolor='black')
plt.xlabel('Weight Value')
plt.ylabel('Frequency')
plt.show()

def relu(Z):
    """
    Applies the ReLU activation function.

    Parameters:
    Z -- A numpy array of any shape.

    Returns:
    A -- Output of ReLU(Z), same shape as Z.
    """
    return np.maximum(0, Z)

def relu_derivative(Z):
    """
    Computes the derivative of the ReLU function.

    Parameters:
    Z -- A numpy array of any shape.

    Returns:
    dZ -- Gradient of ReLU(Z), same shape as Z.
    """
    dZ = np.zeros_like(Z)
    dZ[Z > 0] = 1
    return dZ

def softmax(Z):
    """
    Applies the Softmax function to an input array.

    Parameters:
    Z -- A numpy array of shape (n, m) where n is the number of classes and m is the number of examples.

    Returns:
    A -- Softmax probabilities for each class, same shape as Z.
    """
    # Subtracting the max value of Z for numerical stability
    Z_shifted = Z - np.max(Z, axis=0, keepdims=True)

    # Compute the exponential of all elements in the shifted input
    exp_Z = np.exp(Z_shifted)

    # Compute the softmax probabilities
    softmax_probs = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    return softmax_probs

def forward_prop(w1, b1, w2, b2, X):
  Z1 = w1.dot(X)+b1
  A1 = relu(Z1)
  Z2 = w2.dot(A1)+b2
  A2 = softmax(Z2)
  return Z1, A1, Z2, A2

def one_hot_encoding(Y_train, num_classes):
    num_samples = len(Y_train)
    one_hot_matrix = np.zeros((num_samples, num_classes))

    for i in range(num_samples):
        class_index = int(Y_train[i])
        one_hot_matrix[i,class_index] = 1

    return one_hot_matrix

def back_prop(w1, b1, w2, b2, X, Y, Z1, A1, Z2, A2):
    """
    Performs backpropagation to compute gradients.

    Parameters:
    w1, b1 -- Weights and biases for the first layer
    w2, b2 -- Weights and biases for the second layer
    X -- Input data (features)
    Y -- True labels (one-hot encoded)
    Z1 -- Linear output of the first layer (before activation)
    A1 -- Activation output of the first layer (after ReLU)
    Z2 -- Linear output of the second layer (before softmax)
    A2 -- Predicted output (after softmax)

    Returns:
    dw1, db1 -- Gradients of w1 and b1
    dw2, db2 -- Gradients of w2 and b2
    """

    m = X.shape[1]  # Number of examples

    # Compute the gradient of the loss with respect to Z2 (softmax output)
    dZ2 = A2 - Y

    # Gradients for the second layer (W2, b2)
    dw2 = 1/m * dZ2.dot(A1.T)
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)

    # Compute the gradient of the loss with respect to A1 (activation of the first layer)
    dA1 = w2.T.dot(dZ2)

    # Compute the gradient of the loss with respect to Z1 (before ReLU)
    dZ1 = dA1 * relu_derivative(Z1)

    # Gradients for the first layer (W1, b1)
    dw1 = 1/m * dZ1.dot(X.T)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)

    return dw1, db1, dw2, db2

numbers = np.linspace(-10, 10, 200)   # 200 means 200 values here.
plt.plot(numbers,relu(numbers))
plt.show()

def update_parameters(W1, b1, W2, b2,W3,b3, dW1, db1, dW2, db2,dW3,db3, learning_rate):
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    W3 = W3 - learning_rate * dW3
    b3 = b3 - learning_rate * db3

    return W1, b1, W2, b2,W3,b3

def calculate_accuracy(predictions, Y_train):
    num_samples = Y_train.shape[1]
    actual_labels = np.argmax(Y_train, axis=0)
    num_correct = np.sum(predictions == actual_labels)
    accuracy = num_correct / num_samples
    return accuracy

print("Y_train shape:", Y_train.shape)
print("Y_train contents:", Y_train)

predicted_labels = np.argmax(Y_train, axis=0)
predicted_labels.shape

def get_predictions(output_vector):
    max_index = np.argmax(output_vector, axis=1)
    return max_index

import numpy as np

def gradient_descent(X_train, Y_train, learning_rate=0.01, num_iterations=1000):
    """
    Perform gradient descent to fit a linear model.

    Parameters:
    - X: numpy array of shape (n_samples, n_features), feature matrix.
    - y: numpy array of shape (n_samples,), target vector.
    - learning_rate: float, the learning rate for gradient descent.
    - num_iterations: int, number of iterations to perform.

    Returns:
    - theta: numpy array of shape (n_features,), the optimized parameters.
    - history: list, the history of the cost function values during optimization.
    """
    n_samples, n_features = X.shape
    # Initialize parameters
    theta = np.zeros(n_features)
    history = []

    for i in range(num_iterations):
        # Compute predictions
        predictions = X.dot(theta)
        # Compute the error
        error = predictions - y
        # Compute the gradient
        gradient = (1 / n_samples) * X.T.dot(error)
        # Update parameters
        theta -= learning_rate * gradient
        # Compute and record the cost
        cost = (1 / (2 * n_samples)) * np.sum(error ** 2)
        history.append(cost)

    return theta, history

def gradient_descent(X, y, learning_rate=0.01, num_iterations=1000, n_hidden=10):
    """
    Perform gradient descent to train a neural network.

    Parameters:
    - X: numpy array of shape (n_samples, n_features), feature matrix.
    - y: numpy array of shape (n_samples,), target vector.
    - learning_rate: float, the learning rate for gradient descent.
    - num_iterations: int, number of iterations to perform.
    - n_hidden: int, number of units in the hidden layer.

    Returns:
    - w1: weights for input to hidden layer.
    - b1: biases for hidden layer.
    - w2: weights for hidden to output layer.
    - b2: biases for output layer.
    - predictions: final predictions after training.
    """
    n_features = X.shape[1]
    n_output = 1  # Binary classification

    # Initialize parameters
    w1, b1, w2, b2 = initialize_weights(n_features, n_hidden, n_output)

    # Lists to store the parameters
    parameters_history = []
    accuracy_history = []

    for i in range(num_iterations):
        # Forward propagation
        predictions, cache = forward_prop(X, w1, b1, w2, b2)

        # Backward propagation
        grads = back_prop(X, y, cache, w2)

        # Update parameters
        w1, b1, w2, b2 = update_parameters(w1, b1, w2, b2, grads, learning_rate)

        # Every 5th iteration, print accuracy
        if i % 5 == 0:
            accuracy = calculate_accuracy(predictions, y)
            accuracy_history.append(accuracy)
            print(f"Iteration {i}: Accuracy = {accuracy:.4f}")

        # Store parameters
        parameters_history.append((i, w1.copy(), b1.copy(), w2.copy(), b2.copy()))

    return parameters_history, predictions

def make_prediction(test_vector, W1, b1, W2, b2, index,alpha):
    # Assuming you have functions to preprocess test_vector if needed
    num_classes= b2.shape[0]
    # Forward Propagation
    Z1, A1, Z2, A2  = forward_prop(test_vector, W1, b1, W2, b2,alpha)

    # Obtain Predictions
    A2_reshaped = A2.T
    predictions = get_predictions(A2_reshaped)

    return predictions

Z1, A1, Z2, A2 = forward_prop(w1, b1, w2, b2, X_test)

A2_reshaped = A2.T
predictions = get_predictions(A2_reshaped)

num_samples = Y_test.shape[0]  # Access the first (and only) element of the shape tuple
actual_labels = np.argmax(Y_test, axis=0)  # This line and the following should work fine now
num_correct = np.sum(predictions == actual_labels)
accuracy = num_correct / num_samples

accuracy

print(f"Y_test shape: {Y_test.shape}")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# Calculate accuracy
num_samples = Y_test.shape[0]  # Number of samples
actual_labels = Y_test  # True class indices
num_correct = np.sum(predictions == actual_labels)  # Count correct predictions
accuracy = num_correct / num_samples
print(f"Accuracy: {accuracy:.2f}")

# Convert predicted class indices to one-hot encoded probabilities
num_classes = len(np.unique(Y_test))  # Number of unique classes
predicted_probabilities = np.zeros((num_samples, num_classes))
predicted_probabilities[np.arange(num_samples), predictions] = 1  # One-hot encoding

# Plot ROC curves for each class
plt.figure()
for class_index in range(num_classes):
    # Binary labels for the current class (one-vs-rest approach)
    binary_actual = (actual_labels == class_index).astype(int)
    binary_predicted = predicted_probabilities[:, class_index]

    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(binary_actual, binary_predicted)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.plot(fpr, tpr, lw=2, label=f'Class {class_index} (AUC = {roc_auc:.2f})')

# Plot settings
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

