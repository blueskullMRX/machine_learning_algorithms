# A Random Forest is an ensemble of decision trees. We create multiple decision trees,
# each with a random subset of features and a random subset of samples. This helps
# prevent overfitting by reducing the correlation between trees.

import numpy as np
import pandas as pd
from typing import List
from models.DT import fit, predict

class RandomForest:
    def __init__(self, n_estimators=10, max_depth=None, max_features=None, verbose=False):
        """
        Random Forest constructor.
        :param n_estimators: Number of trees in the forest.
        :param max_depth: Maximum depth of each tree.
        :param max_features: Maximum number of features to consider when splitting.
        :param verbose: Verbosity level (True/False).
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.verbose = verbose
        self.trees = []

    def _bootstrap_sample(self, X, y):
        """Generate a bootstrap sample from the data."""
        if self.verbose:
            print("Bootstrap sampling...")
        indices = np.random.choice(len(X), size=len(X), replace=True)
        return X.iloc[indices], y.iloc[indices]

    def _select_features(self, X):
        """Randomly select features for a single tree."""
        if self.verbose:
            print("Selecting features...")
        n_features = X.shape[1]
        max_features = self.max_features or int(np.sqrt(n_features))  # Default: sqrt(n_features)
        selected_features = np.random.choice(X.columns, size=max_features, replace=False)
        return X[selected_features]

    def fit(self, X, y):
        """
        Train the Random Forest by creating and fitting individual decision trees.
        :param X: Feature matrix (DataFrame).
        :param y: Target vector (Series).
        """
        if self.verbose:
            print(f"Training {self.n_estimators} trees...")
        for i in range(self.n_estimators):
            if self.verbose:
                print(f"Training tree {i+1} of {self.n_estimators}...")
            # Step 1: Bootstrap sampling
            X_sample, y_sample = self._bootstrap_sample(X, y)
            
            # Step 2: Feature subsetting
            X_sample = self._select_features(X_sample)

            # Step 3: Train a decision tree
            tree = fit(X_sample, y_sample, max_depth=self.max_depth)
            self.trees.append((tree, X_sample.columns))  # Store tree and selected features

    def _predict_single(self, x):
        """Predict a single instance by aggregating predictions from all trees."""
        predictions = []
        for tree, features in self.trees:
            row = x[features]  # Use only the features selected for this tree
            predictions.append(predict(pd.DataFrame([row]), tree).iloc[0])
        return max(set(predictions), key=predictions.count)  # Majority vote

    def predict(self, X):
        """
        Predict the output for the given feature matrix.
        :param X: Feature matrix (DataFrame).
        :return: Predicted values (Series).
        """
        if self.verbose:
            print("Predicting...")
        return X.apply(self._predict_single, axis=1)

# Example usage
if __name__ == "__main__":
    from seaborn import load_dataset
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    # Load dataset and prepare data
    titanic = load_dataset('titanic')
    titanic.drop(columns=['embarked', 'who', 'deck', 'alive', 'alone', 'adult_male'], inplace=True)
    titanic.dropna(inplace=True)
    titanic.drop_duplicates(inplace=True)

    categorical_features = titanic.select_dtypes(include=['object', 'category'])
    numerical_features = titanic.select_dtypes(include=['float64', 'int64']).drop(columns=['survived'])

    X = pd.concat([categorical_features, numerical_features], axis=1)
    y = titanic['survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train random forest
    rf = RandomForest(n_estimators=10, max_depth=5)
    print("Training...", end="")
    rf.fit(X_train, y_train)
    print("Done")

    # Predict and evaluate
    y_pred = rf.predict(X_test)
    print("Evaluating...")
    print(classification_report(y_test, y_pred))

