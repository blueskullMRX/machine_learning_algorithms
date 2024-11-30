import numpy as np
import pandas as pd


class DecisionTree:
    def __init__(self, min_data=10, max_depth=5, target_feature='class'):
        self.min_data = min_data
        self.max_depth = max_depth
        self.target_feature = target_feature
        self.tree = None

    @staticmethod
    def entropy(dataset, feature_name, target_name):
        feature = dataset[feature_name].to_numpy()
        target = dataset[target_name].to_numpy()

        unique_features = np.unique(feature)
        unique_targets = np.unique(target)

        counts = np.zeros((len(unique_features), len(unique_targets)), dtype=int)
        for i, f_val in enumerate(unique_features):
            for j, t_val in enumerate(unique_targets):
                counts[i, j] = np.sum((feature == f_val) & (target == t_val))

        total_entropy = 0
        data_size = dataset.shape[0]
        for i in range(counts.shape[0]):
            total_count = sum(counts[i])
            entropy_each_value = 0
            for j in range(counts.shape[1]):
                p = counts[i][j] / total_count
                if p != 0:
                    entropy_each_value += -p * np.log2(p)

            feature_value_count = dataset[dataset[feature_name] == unique_features[i]].shape[0]
            total_entropy += feature_value_count / data_size * entropy_each_value

        return total_entropy

    @staticmethod
    def entropy_main(dataset, target_name):
        target = dataset[target_name].to_numpy()
        unique_targets, counts = np.unique(target, return_counts=True)

        total = sum(counts)
        entropy = 0
        for i in range(len(unique_targets)):
            p = counts[i] / total
            entropy += -p * np.log2(p)

        return entropy

    def gain(self, dataset, feature, target):
        return self.entropy_main(dataset, target) - self.entropy(dataset, feature, target)

    @staticmethod
    def si(dataset, feature_name):
        unique_features, counts = np.unique(dataset[feature_name], return_counts=True)
        total = sum(counts)
        si = 0
        for i in range(len(unique_features)):
            p = counts[i] / total
            si += -p * np.log2(p)
        return si

    def gain_ratio(self, dataset, feature, target):
        gn = self.gain(dataset, feature, target)
        s = self.si(dataset, feature)
        if s == 0:
            return 0
        return gn / s

    def find_highest_gain_ratio_value(self, data):
        features = data.columns.tolist()
        features.remove(self.target_feature)

        max_gain = -1
        max_gain_feature = None
        for feature in features:
            gain = self.gain_ratio(data, feature, self.target_feature)
            if gain > max_gain:
                max_gain = gain
                max_gain_feature = feature
        return max_gain_feature

    @staticmethod
    def divide_data(data, feature):
        feature_values = data[feature].unique()
        datas = {}
        for feature_value in feature_values:
            data_byvalue = data[data[feature] == feature_value]
            data_byvalue = data_byvalue.drop(feature, axis='columns')
            datas[feature_value] = data_byvalue
        return datas

    def build_tree(self, data, current_depth=0):
        # Stopping conditions
        if data.shape[0] < self.min_data or len(data[self.target_feature].unique()) == 1 or current_depth >= self.max_depth:
            return data[self.target_feature].mode()[0]

        # Best feature selection
        best_feature = self.find_highest_gain_ratio_value(data)
        if best_feature is None:
            return data[self.target_feature].mode()[0]

        tree = {best_feature: {}}
        for value, subset in self.divide_data(data, best_feature).items():
            subtree = self.build_tree(subset, current_depth + 1)
            tree[best_feature][value] = subtree

        return tree

    def fit(self, data):
        self.tree = self.build_tree(data)

    def classify(self, tree, sample):
        if not isinstance(tree, dict):
            return tree

        feature = next(iter(tree))
        value = sample[feature]
        if value not in tree[feature]:
            return None
        subtree = tree[feature][value]
        return self.classify(subtree, sample)

    def predict(self, data):
        return data.apply(lambda row: self.classify(self.tree, row), axis=1)

    @staticmethod
    def accuracy(predicted, result):
        return (predicted == result).mean()


# Test Case
if __name__ == "__main__":
    # Sample dataset
    data = pd.DataFrame({
        'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
        'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
        'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
        'Windy': [False, True, False, False, False, True, True, False, False, False, True, True, False, True],
        'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
    })

    # Instantiate and train the decision tree
    dt = DecisionTree(target_feature='Play', max_depth=3)
    dt.fit(data)

    # Make predictions
    predictions = dt.predict(data)

    # Calculate accuracy
    accuracy_score = dt.accuracy(predictions, data['Play'])

    print("Decision Tree Structure:")
    print(dt.tree)
    print("\nPredictions:")
    print(predictions.tolist())
    print(f"\nAccuracy: {accuracy_score * 100:.2f}%")
