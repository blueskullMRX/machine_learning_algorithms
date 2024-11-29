import numpy as np

class LightGBMClassifier:
    """
    A simple implementation of a LightGBM-like gradient boosting classifier.
    Currently supports binary classification.
    """
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        """
        Initialize the LightGBM classifier.

        Parameters:
        - n_estimators (int): Number of boosting rounds.
        - learning_rate (float): Step size shrinkage used to prevent overfitting.
        - max_depth (int): Maximum depth of each tree (unused in this implementation).
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    @staticmethod
    def _sigmoid(x):
        """Compute the sigmoid of x."""
        return 1 / (1 + np.exp(-x))

    def _compute_leaf_value(self, gradients, hessians):
        """Compute the optimal leaf value for a single leaf."""
        return -np.sum(gradients) / (np.sum(hessians) + 1e-6)

    def _build_tree(self, X, gradients, hessians):
        """
        Build a simple tree for this iteration.
        Currently, it produces a single-leaf tree.
        """
        leaf_value = self._compute_leaf_value(gradients, hessians)
        return lambda x: np.full(x.shape[0], leaf_value)

    def _validate_input(self, X, y):
        """Ensure input arrays are valid."""
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y must be numpy arrays.")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples.")
        if len(np.unique(y)) != 2:
            raise ValueError("Only binary classification is supported.")

    def fit(self, X, y):
        """
        Fit the model to the training data.

        Parameters:
        - X (numpy.ndarray): Training features of shape (n_samples, n_features).
        - y (numpy.ndarray): Binary labels (0/1 or -1/1) of shape (n_samples,).
        """
        self._validate_input(X, y)
        m, n = X.shape
        predictions = np.zeros(m)
        
        # Ensure binary labels are -1 and 1
        unique_labels = np.unique(y)
        y_encoded = np.where(y == unique_labels[0], -1, 1)

        for _ in range(self.n_estimators):
            # Compute gradients and hessians
            sigmoid_preds = self._sigmoid(predictions)
            gradients = -y_encoded * (1 - sigmoid_preds)
            hessians = sigmoid_preds * (1 - sigmoid_preds)

            # Build and store the current tree
            tree = self._build_tree(X, gradients, hessians)
            self.trees.append(tree)

            # Update predictions
            predictions += self.learning_rate * tree(X)

    def predict(self, X):
        """
        Predict binary labels for the input data.

        Parameters:
        - X (numpy.ndarray): Input features of shape (n_samples, n_features).

        Returns:
        - numpy.ndarray: Predicted binary labels (0/1).
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy array.")
        
        predictions = np.zeros(X.shape[0])
        for tree in self.trees:
            predictions += self.learning_rate * tree(X)
        return (predictions > 0).astype(int)

