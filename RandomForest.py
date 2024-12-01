from DecisionTreeC45 import DecisionTree
import numpy as np
import pandas as pd


class RandomForest:
    """
    Random Forest implementation using Decision Trees with random feature selection.
    """

    def __init__(self, n_trees=5, max_depth=4, min_data=10, target_feature="class"):
        """
        Initialize the Random Forest model.

        Parameters:
        - n_trees (int): Number of decision trees in the forest.
        - max_depth (int): Maximum depth of each tree.
        - min_data (int): Minimum samples required to split a node.
        - target_feature (str): The target feature for classification.
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_data = min_data
        self.target_feature = target_feature
        self.trees = []
        self.oob_data = None

    def _random_feature_selection(self, features):
        """
        Randomly select a subset of features.

        Parameters:
        - features (list): List of feature names.

        Returns:
        - list: Subset of randomly selected features.
        """
        num_features_to_select = int(np.ceil(np.sqrt(len(features))))
        return np.random.choice(features, size=num_features_to_select, replace=False).tolist()

    def _bootstrap_sample(self, data):
        """
        Create a bootstrap sample and its corresponding out-of-bag data.

        Parameters:
        - data (pd.DataFrame): Input data.

        Returns:
        - tuple: (bootstrap sample, out-of-bag data).
        """
        n = len(data)
        bootstrap_indices = np.random.choice(range(n), size=n, replace=True)
        oob_indices = list(set(range(n)) - set(bootstrap_indices))
        return data.iloc[bootstrap_indices], data.iloc[oob_indices]

    def _build_tree(self, data):
        """
        Build a decision tree for the Random Forest using random feature selection.

        Parameters:
        - data (pd.DataFrame): Training data.

        Returns:
        - DecisionTree: Trained DecisionTree instance.
        """
        features = data.columns.tolist()
        features.remove(self.target_feature)
        selected_features = self._random_feature_selection(features)

        # Include the target feature in the subset of data for training
        selected_features.append(self.target_feature)
        selected_data = data[selected_features]

        tree = DecisionTree(
            max_depth=self.max_depth,
            min_data=self.min_data,
            target_feature=self.target_feature,
        )
        tree.fit(selected_data)
        return tree

    def fit(self, data):
        """
        Train the Random Forest model.

        Parameters:
        - data (pd.DataFrame): Training data.
        """
        self.trees = []
        oob_data_list = []

        for _ in range(self.n_trees):
            bootstrap, oob = self._bootstrap_sample(data)
            oob_data_list.append(oob)
            self.trees.append(self._build_tree(bootstrap))

        self.oob_data = pd.concat(oob_data_list).drop_duplicates().reset_index(drop=True)

    def predict(self, data):
        """
        Predict class labels for input data.

        Parameters:
        - data (pd.DataFrame): Input data.

        Returns:
        - np.ndarray: Predicted class labels.
        """
        predictions = [tree.predict(data) for tree in self.trees]
        predictions_df = pd.DataFrame(predictions).T
        return predictions_df.mode(axis=1)[0].values

    def evaluate_oob(self):
        """
        Evaluate the model using out-of-bag data.

        Returns:
        - float: Accuracy on out-of-bag data.
        """
        if self.oob_data is None:
            raise ValueError("Out-of-bag data not available. Train the model first.")

        y_true = self.oob_data[self.target_feature].values
        y_pred = self.predict(self.oob_data)
        return (y_true == y_pred).mean()

    def accuracy(self, y_pred, y_true):
        return (y_true == y_pred).mean()

# Example usage
if __name__ == "__main__":
    data = pd.DataFrame({
        'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
        'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
        'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
        'Windy': [False, True, False, False, False, True, True, False, False, False, True, True, False, True],
        'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
    })

    # Instantiate and train the decision tree
    rf = RandomForest(n_trees=5, max_depth=3, min_data=10, target_feature='Play')
    rf.fit(data)
    predictions = rf.predict(data)
    accuracy_score = rf.accuracy(predictions, data['Play'])

    print(f"\nAccuracy: {accuracy_score * 100:.2f}%")
