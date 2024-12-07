import numpy as np
import pandas as pd

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))
    

def compute_loss(y, y_pred):
    m = len(y)
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    return -1 / m * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

def train_lr(X,y,learning_rate=0.01,epsilon=0.00001,max_iteration = 3000):
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0
    loss = None

    for iteration in range(max_iteration):
        prev_loss = loss

        z = np.dot(X, weights) + bias

        y_pred = sigmoid(z)

        dw = (1 / m) * np.dot(X.T, (y_pred - y))
        db = (1 / m) * np.sum(y_pred - y)

        weights -= learning_rate * dw
        bias -= learning_rate * db

        loss = compute_loss(y, y_pred)

        if prev_loss != None and np.abs(loss-prev_loss) < epsilon:
            #print(f"Converged at iteration {iteration + 1}")
            break
    return weights,bias

def predict_proba(X,weights,bias):
    z = np.dot(X, weights) + bias
    return sigmoid(z)

def predict(X,weights,bias):
    p = predict_proba(X,weights,bias)
    return (p >= 0.5).astype(int)
