import numpy as np

class LogisticRegression:
    """
    Logistic Regression model trained using gradient descent.
    Supports binary classification.
    """
    def __init__(self, learning_rate=0.01, epsilon=1e-5, max_iter=3000):
        """
        Initialize the model.

        Parameters:
        - learning_rate (float): Learning rate for gradient descent.
        - epsilon (float): Convergence threshold.
        - max_iter (int): Maximum number of iterations for training.
        """
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.weights = None
        self.bias = 0

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """
        Numerically stable sigmoid function to prevent overflow and handle invalid division.
        
        Args:
            x (np.ndarray): Input array
        
        Returns:
            np.ndarray: Sigmoid of input
        """
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            positive_mask = x >= 0
            negative_mask = ~positive_mask
            
            z = np.zeros_like(x, dtype=float)
            z[positive_mask] = 1 / (1 + np.exp(-x[positive_mask]))
            z[negative_mask] = np.exp(x[negative_mask]) / (1 + np.exp(x[negative_mask]))
            
            return z

    @staticmethod
    def compute_loss(y, y_pred):
        """
        Compute the logistic regression loss (cross-entropy loss).

        Parameters:
        - y (numpy.ndarray): True labels.
        - y_pred (numpy.ndarray): Predicted probabilities.

        Returns:
        - float: Computed loss.
        """
        return -np.mean(y * np.log(y_pred + 1e-10) + (1 - y) * np.log(1 - y_pred + 1e-10))

    def fit(self, X, y):
        """
        Train the logistic regression model using gradient descent.

        Parameters:
        - X (numpy.ndarray): Training features of shape (n_samples, n_features).
        - y (numpy.ndarray): Training labels of shape (n_samples,).
        """
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for iteration in range(self.max_iter):
            # Compute predictions
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)

            # Compute gradients
            dw = np.dot(X.T, (y_pred - y)) / m
            db = np.mean(y_pred - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Compute loss and check for convergence
            loss = self.compute_loss(y, y_pred)
            if iteration > 0 and np.abs(loss - prev_loss) < self.epsilon:
                print(f"Converged at iteration {iteration + 1}")
                break
            prev_loss = loss

    def predict_proba(self, X):
        """
        Predict the probabilities of the positive class for input features.

        Parameters:
        - X (numpy.ndarray): Input features of shape (n_samples, n_features).

        Returns:
        - numpy.ndarray: Predicted probabilities.
        """
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    def predict(self, X):
        """
        Predict binary labels for input features.

        Parameters:
        - X (numpy.ndarray): Input features of shape (n_samples, n_features).

        Returns:
        - numpy.ndarray: Predicted binary labels (0 or 1).
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)

# Example Usage
if __name__ == "__main__":
    np.random.seed(0)
    X = np.random.rand(100, 2)
    y = (X[:, 0] + X[:, 1] >= 1).astype(int)

    model = LogisticRegression(learning_rate=0.1, epsilon=1e-5, max_iter=3000)
    model.fit(X, y)
    y_pred = model.predict(X)

    print("Predictions:", y_pred)
