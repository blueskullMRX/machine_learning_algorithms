import numpy as np


class LightGBM:
    """
    Simplified LightGBM-like gradient boosting classifier for binary classification.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1):
        """
        Parameters:
        - n_estimators (int): Number of boosting rounds.
        - learning_rate (float): Step size shrinkage to control overfitting.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.trees = []

    @staticmethod
    def sigmoid(x):
        """Compute the sigmoid function."""
        return 1 / (1 + np.exp(-x))

    def _validate_data(self, X, y):
        """Ensure data is valid for training."""
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("Both X and y must be numpy arrays.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        if np.unique(y).size != 2:
            raise ValueError("This implementation only supports binary classification.")

    def _leaf_value(self, gradients, hessians):
        """Compute the leaf value."""
        return -np.sum(gradients) / (np.sum(hessians) + 1e-6)

    def _build_tree(self, X, gradients, hessians):
        """Construct a tree, represented as a single-leaf constant prediction."""
        leaf_value = self._leaf_value(gradients, hessians)
        return lambda x: np.full(x.shape[0], leaf_value)

    def fit(self, X, y):
        """
        Fit the model.

        Parameters:
        - X (numpy.ndarray): Features (n_samples, n_features).
        - y (numpy.ndarray): Binary labels (0/1 or -1/1) (n_samples,).
        """
        self._validate_data(X, y)
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)

        # Map labels to -1 and 1 for gradient boosting
        y_mapped = np.where(y == 0, -1, 1)

        for _ in range(self.n_estimators):
            # Compute gradients and hessians
            sigmoid_preds = self.sigmoid(predictions)
            gradients = -y_mapped * (1 - sigmoid_preds)
            hessians = sigmoid_preds * (1 - sigmoid_preds)

            # Create a tree and update predictions
            tree = self._build_tree(X, gradients, hessians)
            self.trees.append(tree)
            predictions += self.learning_rate * tree(X)

    def predict(self, X):
        """
        Predict binary labels.

        Parameters:
        - X (numpy.ndarray): Features (n_samples, n_features).

        Returns:
        - numpy.ndarray: Predicted binary labels (0/1).
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy array.")

        if len(self.trees) == 0:
            raise ValueError("Model has not been trained yet. Try running `fit` first.")
        
        predictions = np.sum([self.learning_rate * tree(X) for tree in self.trees], axis=0)
        return (predictions > 0).astype(int)


if __name__ == "__main__":
    # Test case
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LightGBM(n_estimators=20, learning_rate=0.001)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))

