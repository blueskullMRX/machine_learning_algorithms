import numpy as np
import pandas as pd

###############################################################################################################################
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature  # Index of feature to split on
        self.threshold = threshold  # Threshold for the split
        self.left = left  # Left subtree
        self.right = right  # Right subtree
        self.value = value  # Value for leaf node

def build_tree(X, y,max_depth=5,min_samples_split=10, depth=0):
    n_samples, n_features = X.shape
    if depth >= max_depth or n_samples < min_samples_split or np.all(y == y[0]):
        return Node(value=np.mean(y))

    best_split = find_best_split(X, y,min_samples_split)
    if not best_split:
        return Node(value=np.mean(y))

    left_subtree = build_tree(best_split["X_left"], best_split["y_left"],max_depth,min_samples_split, depth + 1)
    right_subtree = build_tree(best_split["X_right"], best_split["y_right"],max_depth,min_samples_split, depth + 1)

    return Node(feature=best_split["feature"], threshold=best_split["threshold"],left=left_subtree, right=right_subtree)

def find_best_split(X, y,min_samples_split):
    best_mse = float("inf")
    best_split = None
    n_samples, n_features = X.shape

    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            X_left, X_right, y_left, y_right = split(X, y, feature, threshold)
            if len(y_left) < min_samples_split or len(y_right) < min_samples_split:
                continue

            mse = (len(y_left) * mse_calc(y_left) + len(y_right) * mse_calc(y_right)) / n_samples
            if mse < best_mse:
                best_mse = mse
                best_split = {"feature": feature, "threshold": threshold, "X_left": X_left, "X_right": X_right,
                                "y_left": y_left, "y_right": y_right}

    return best_split
    
def split(X, y, feature, threshold):
    if isinstance(X, np.ndarray):
        left_indices = X[:, feature] <= threshold
        right_indices = X[:, feature] > threshold
    elif isinstance(X, pd.DataFrame):  # Pandas DataFrame
        left_indices = X.iloc[:, feature] <= threshold
        right_indices = X.iloc[:, feature] > threshold
    else:
        raise ValueError("X must be a NumPy array or Pandas DataFrame")

    return X[left_indices], X[right_indices], y[left_indices], y[right_indices]
    

def mse_calc(y):
    if len(y) == 0:
        return 0
    return np.mean((y - np.mean(y)) ** 2)

def predict_sample(x, tree):
        """Predict a single sample."""
        if tree.value is not None:
            return tree.value
        if x[tree.feature] <= tree.threshold:
            return predict_sample(x, tree.left)
        return predict_sample(x, tree.right)

def tree_predict(X,tree):
    return np.array([predict_sample(x, tree) for x in X])




############################################################

def fit(X, y,n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2):
    init_prediction = np.mean(y)
    y_pred = np.full(y.shape, init_prediction)
    trees = []
    for _ in range(n_estimators):
        grad, hess = loss_gradient(y, y_pred)

        tree = build_tree(X,grad,max_depth,min_samples_split)

        trees.append(tree)

        y_pred += learning_rate * tree_predict(X,tree)
    return trees,init_prediction,learning_rate

def loss_gradient(y, y_pred):
    grad = y - y_pred
    hess = np.ones_like(y)
    return grad, hess

def predict(X,trees,init_prediction,learning_rate):
    y_pred = np.full((X.shape[0],), init_prediction)
    for tree in trees:
        y_pred += learning_rate * tree_predict(X,tree)
    y_pred = np.where(y_pred > 0.5, 1, 0)
    return y_pred
