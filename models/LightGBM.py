import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from DT import fit, predict

class LightGBMClassifier:
    """
    A simplified LightGBM-like gradient boosting classifier.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.tree_weights = []

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        """
        Train the LightGBM classifier.
        """

        predictions = np.zeros(len(y))
        for _ in range(self.n_estimators):
            residuals = pd.Series(y - self._sigmoid(predictions))

            tree = fit(X, y, max_depth=self.max_depth)
            tree_predictions = predict(X, tree)

            self.trees.append(tree)
            predictions += self.learning_rate * tree_predictions

    def predict(self, X):
        X = X.values if isinstance(X, pd.DataFrame) else X

        predictions = np.zeros(X.shape[0])
        for tree in self.trees:
            predictions += self.learning_rate * predict(X)

        probabilities = self._sigmoid(predictions)
        return (probabilities > 0.5).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


def main():
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target

    X = X.dropna().drop_duplicates()
    y = y.loc[X.index].reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LightGBMClassifier(n_estimators=10, learning_rate=0.01, max_depth=3)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
