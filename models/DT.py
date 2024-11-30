import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Union
from seaborn import load_dataset
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize_scalar

def shannon_entropy(labels):
    """Calculate the Shannon entropy of the given labels."""
    label_counts = labels.value_counts(normalize=True)
    return -np.sum([p * np.log2(p) for p in label_counts if p > 0])

def find_numeric_thresholds(feature_values, labels):
    """Find the best threshold to split numeric features for decision tree."""
    sorted_values = feature_values.sort_values().unique()
    best_threshold = None
    best_info_gain = 0
    base_entropy = shannon_entropy(labels)

    for i in range(1, len(sorted_values)):
        threshold = (sorted_values[i - 1] + sorted_values[i]) / 2
        left_mask = feature_values <= threshold
        right_mask = feature_values > threshold

        if left_mask.sum() == 0 or right_mask.sum() == 0:
            continue

        left_entropy = shannon_entropy(labels[left_mask])
        right_entropy = shannon_entropy(labels[right_mask])

        weighted_entropy = (
            left_mask.mean() * left_entropy + right_mask.mean() * right_entropy
        )
        info_gain = base_entropy - weighted_entropy

        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_threshold = threshold

    return best_threshold, best_info_gain

def find_numeric_threshold_global(feature_values, labels):
    """Find the global best threshold using optimization for numeric features."""
    sorted_values = feature_values.sort_values().unique()
    base_entropy = shannon_entropy(labels)

    def entropy_split(threshold):
        left_mask = feature_values <= threshold
        right_mask = feature_values > threshold
        left_entropy = shannon_entropy(labels[left_mask])
        right_entropy = shannon_entropy(labels[right_mask])
        return (
            left_mask.mean() * left_entropy + right_mask.mean() * right_entropy
        )

    result = minimize_scalar(
        entropy_split,
        bounds=(sorted_values[0], sorted_values[-1]),
        method='bounded',
        options={'xatol': 1e-6}
    )

    best_threshold = result.x
    best_info_gain = base_entropy - result.fun

    return best_threshold, best_info_gain

def find_best_feature(data, labels):
    """Identify the best feature to split data on."""
    best_feature = None
    best_threshold = None
    best_info_gain = 0

    for feature in data.columns:
        feature_values = data[feature]
        if pd.api.types.is_numeric_dtype(feature_values):
            threshold, info_gain = find_numeric_threshold_global(feature_values, labels)
        else:
            unique_values = feature_values.dropna().unique()
            base_entropy = shannon_entropy(labels)

            weighted_entropy = sum(
                (feature_values == value).mean() *
                shannon_entropy(labels[feature_values == value])
                for value in unique_values
            )
            info_gain = base_entropy - weighted_entropy
            threshold = None

        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = feature
            best_threshold = threshold

    return best_feature, best_threshold

def create_tree(data, labels, depth=0, max_depth=None):
    """Recursively build the decision tree based on features and labels."""
    if (max_depth is not None and depth >= max_depth) or len(labels.unique()) == 1:
        return labels.mode()[0]

    best_feature, best_threshold = find_best_feature(data, labels)
    if best_feature is None:
        return labels.mode()[0]

    tree = {best_feature: {}}

    if best_threshold is not None:
        left_mask = data[best_feature] <= best_threshold
        right_mask = data[best_feature] > best_threshold

        tree[best_feature][f"≤{best_threshold}"] = create_tree(
            data[left_mask].drop(columns=[best_feature]),
            labels[left_mask],
            depth + 1,
            max_depth,
        )
        tree[best_feature][f">{best_threshold}"] = create_tree(
            data[right_mask].drop(columns=[best_feature]),
            labels[right_mask],
            depth + 1,
            max_depth,
        )
    else:
        for value in data[best_feature].dropna().unique():
            mask = data[best_feature] == value
            tree[best_feature][value] = create_tree(
                data[mask].drop(columns=[best_feature]),
                labels[mask],
                depth + 1,
                max_depth,
            )

    return tree

def fit(data, labels, max_depth=None):
    """Fit the decision tree model on the given data and labels."""
    if not isinstance(data, pd.DataFrame) or not isinstance(labels, pd.Series):
        raise TypeError("Data must be a pandas DataFrame and labels must be a pandas Series.")
    return create_tree(data, labels, max_depth=max_depth)

def predict(data, tree):
    """Predict labels for the given data using the decision tree."""
    if tree is None:
        raise ValueError("Model not trained. Call fit() first.")

    predictions = []
    for _, row in data.iterrows():
        node = tree
        while isinstance(node, dict):
            feature = list(node.keys())[0]
            if isinstance(node[feature], dict) and any('≤' in str(k) or '>' in str(k) for k in node[feature].keys()):
                for threshold_key, subtree in node[feature].items():
                    threshold = float(threshold_key[1:])
                    if threshold_key.startswith('≤'):
                        if pd.to_numeric(row[feature], errors='coerce') <= threshold:
                            node = subtree
                            break
                    else:
                        if pd.to_numeric(row[feature], errors='coerce') > threshold:
                            node = subtree
                            break
            else:
                value = row[feature]
                node = node[feature].get(value)

            if not isinstance(node, dict):
                break

        predictions.append(node)

    return pd.Series(predictions)

def print_tree(tree):
    """Print the structure of the decision tree."""
    def recurse(subtree, indent=0):
        if isinstance(subtree, dict):
            for key, subsubtree in subtree.items():
                print("-" * indent + f'> {str(key)}')
                recurse(subsubtree, indent + 1)
        else:
            print(" " * indent + str(subtree))

    recurse(tree)
def main():
    """Main function to load data, train the model, and evaluate its performance."""
    titanic_data = load_dataset('titanic')
    titanic_data.drop(columns=['embarked', 'who', 'deck', 'alive', 'alone', 'adult_male'], inplace=True)
    titanic_data.dropna(inplace=True)
    titanic_data.drop_duplicates(inplace=True)
    categorical_features = titanic_data.select_dtypes(include=['object', 'category'])
    numeric_features = titanic_data.select_dtypes(include=['float64', 'int64']).drop(columns=['survived'])
    data = pd.concat([categorical_features, numeric_features], axis=1)
    labels = titanic_data['survived']
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    decision_tree = fit(X_train, y_train)
    print_tree(decision_tree)
    y_pred = predict(X_test, decision_tree)
    print(classification_report(y_test, y_pred))
if __name__ == "__main__":
    main()
