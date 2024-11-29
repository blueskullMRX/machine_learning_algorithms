import numpy as np
import pandas as pd

def sigmoid(z):
    """Compute the sigmoid of z."""
    return 1 / (1 + np.exp(-z))

def compute_loss(y, y_pred):
    """
    Compute the logistic regression loss (cross-entropy loss).

    Parameters:
    - y (numpy.ndarray): True labels.
    - y_pred (numpy.ndarray): Predicted probabilities.

    Returns:
    - float: Computed loss.
    """
    m = len(y)
    return -1 / m * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

def train_lr(X, y, learning_rate=0.01, epsilon=0.00001, max_iteration=3000):
    """
    Train a logistic regression model using gradient descent.

    Parameters:
    - X (numpy.ndarray): Training features.
    - y (numpy.ndarray): Training labels.
    - learning_rate (float): Learning rate for gradient descent.
    - epsilon (float): Convergence threshold.
    - max_iteration (int): Maximum number of iterations.

    Returns:
    - tuple: Weights and bias of the trained model.
    """
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0
    loss = None

    for iteration in range(max_iteration):
        prev_loss = loss

        # Compute the linear combination and apply sigmoid
        z = np.dot(X, weights) + bias
        y_pred = sigmoid(z)

        # Compute gradients
        dw = (1 / m) * np.dot(X.T, (y_pred - y))
        db = (1 / m) * np.sum(y_pred - y)

        # Update parameters
        weights -= learning_rate * dw
        bias -= learning_rate * db

        # Compute the loss
        loss = compute_loss(y, y_pred)

        # Check for convergence
        if prev_loss is not None and np.abs(loss - prev_loss) < epsilon:
            print(f"Converged at iteration {iteration + 1}")
            break

    return weights, bias

def predict_proba(X, weights, bias):
    """
    Predict the probabilities of the positive class for input features.

    Parameters:
    - X (numpy.ndarray): Input features.
    - weights (numpy.ndarray): Model weights.
    - bias (float): Model bias.

    Returns:
    - numpy.ndarray: Predicted probabilities.
    """
    z = np.dot(X, weights) + bias
    return sigmoid(z)

def predict(X, weights, bias):
    """
    Predict binary labels for input features.

    Parameters:
    - X (numpy.ndarray): Input features.
    - weights (numpy.ndarray): Model weights.
    - bias (float): Model bias.

    Returns:
    - numpy.ndarray: Predicted binary labels (0 or 1).
    """
    p = predict_proba(X, weights, bias)
    return (p >= 0.5).astype(int)
