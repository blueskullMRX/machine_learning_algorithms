import numpy as np
import pandas as pd

def entropy(dataset, feature_name, target_name):
    """
    Calculate the entropy of a dataset for a given feature.

    Parameters
    ----------
    dataset : pandas.DataFrame
        Input dataframe.
    feature_name : str
        Feature for which to compute entropy.
    target_name : str
        Target feature.

    Returns
    -------
    float
        Calculated entropy.
    """
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

def entropy_main(dataset, target_name):
    """
    Calculate the main entropy of the target in the dataset.

    Parameters
    ----------
    dataset : pandas.DataFrame
        Input dataframe.
    target_name : str
        Target feature.

    Returns
    -------
    float
        Calculated main entropy.
    """
    target = dataset[target_name].to_numpy()
    unique_targets, counts = np.unique(target, return_counts=True)

    total = sum(counts)
    entropy = 0
    for i in range(len(unique_targets)):
        p = counts[i] / total
        entropy += -p * np.log2(p)

    return entropy

def gain(dataset, feature, target):
    """
    Calculate the information gain of a feature.

    Parameters
    ----------
    dataset : pandas.DataFrame
        Input dataframe.
    feature : str
        Feature for which to compute the gain.
    target : str
        Target feature.

    Returns
    -------
    float
        Calculated information gain.
    """
    return entropy_main(dataset, target) - entropy(dataset, feature, target)

def si(dataset, feature_name):
    """
    Calculate the split information for a feature.

    Parameters
    ----------
    dataset : pandas.DataFrame
        Input dataframe.
    feature_name : str
        Feature for which to compute split information.

    Returns
    -------
    float
        Calculated split information.
    """
    unique_features, counts = np.unique(dataset[feature_name], return_counts=True)
    total = sum(counts)
    si = 0
    for i in range(len(unique_features)):
        p = counts[i] / total
        si += -p * np.log2(p)
    return si

def gain_ratio(dataset, feature, target):
    """
    Calculate the gain ratio of a feature.

    Parameters
    ----------
    dataset : pandas.DataFrame
        Input dataframe.
    feature : str
        Feature for which to compute gain ratio.
    target : str
        Target feature.

    Returns
    -------
    float
        Calculated gain ratio.
    """
    gn = gain(dataset, feature, target)
    s = si(dataset, feature)
    if s == 0:
        return 0
    return gn / s

def best_split_value(dataset, feature_name, target):
    """
    Find the best split value for a continuous feature.

    Parameters
    ----------
    dataset : pandas.DataFrame
        Input dataframe.
    feature_name : str
        Continuous feature to evaluate.
    target : str
        Target feature.

    Returns
    -------
    float
        Best split value.
    """
    feature = dataset[feature_name].to_numpy()
    unique_features = np.unique(feature)

    gain_ratio_max = -1
    gain_ratio_max_value = None

    for i in range(len(unique_features)):
        categorize = np.zeros(feature.shape)

        for j, value in enumerate(feature):
            categorize[j] = 1 if value > unique_features[i] else 0

        tg = dataset[target].to_numpy()
        test = np.column_stack((categorize, tg))
        test_data = pd.DataFrame(data=test, columns=['feature', 'target'])

        gain_rat = gain_ratio(test_data, 'feature', 'target')
        if gain_rat > gain_ratio_max:
            gain_ratio_max = gain_rat
            gain_ratio_max_value = unique_features[i]

    return gain_ratio_max_value

def to_category(data, feature, target):
    """
    Convert a continuous feature to a categorical feature based on the best split value.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataframe.
    feature : str
        Continuous feature to convert.
    target : str
        Target feature.

    Returns
    -------
    pandas.Series
        Categorical feature.
    """
    best = best_split_value(data, feature, target)
    return data[feature].apply(lambda x: 1 if x > best else 0)

def detect_continuous_columns(df, threshold=10):
    """
    Detect continuous columns in a dataframe based on a threshold.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    threshold : int, optional
        Unique values threshold to detect continuous columns, by default 10.

    Returns
    -------
    list
        List of continuous columns.
    """
    continuous_columns = []
    for column in df.columns:
        if np.issubdtype(df[column].dtype, np.number):
            unique_values = df[column].nunique()
            total_values = len(df[column])

            if unique_values > threshold or unique_values / total_values > 0.5:
                continuous_columns.append(column)
    return continuous_columns

def find_highest_gain_ratio_value(data, target_column):
    """
    Find the feature with the highest gain ratio.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataframe.
    target_column : str
        Target feature.

    Returns
    -------
    str
        Feature with the highest gain ratio.
    """
    features = data.columns.tolist()
    features.remove(target_column)

    max_gain = -1
    max_gain_feature = None

    for feature in features:
        gain = gain_ratio(data, feature, target_column)
        if gain > max_gain:
            max_gain = gain
            max_gain_feature = feature
    return max_gain_feature

def divide_data(data, feature):
    """
    Divide the dataset based on unique values of a feature.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataframe.
    feature : str
        Feature to divide the data on.

    Returns
    -------
    dict
        Dictionary of divided data.
    """
    feature_values = data[feature].unique()
    datas = {}
    for feature_value in feature_values:
        data_byvalue = data[data[feature] == feature_value]
        data_byvalue = data_byvalue.drop(feature, axis='columns')
        datas[feature_value] = data_byvalue
    return datas

def build_tree(data, min_data=10, target_feature='class', max_depth=5, current_depth=0):
    """
    Recursively build a decision tree.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataframe.
    min_data : int, optional
        Minimum data size to split, by default 10.
    target_feature : str, optional
        Target feature, by default 'class'.
    max_depth : int, optional
        Maximum depth of the tree, by default 5.
    current_depth : int, optional
        Current depth of the tree, by default 0.

    Returns
    -------
    dict or str
        Constructed decision tree or the most common target value if stopping condition is met.
    """
    # stopping conditions: data size, no more features to test on or max depth achieved
    if data.shape[0] < min_data or len(data[target_feature].unique()) == 1 or current_depth >= max_depth:
        return data[target_feature].mode()[0]

    # stopping condition: highest chi2 feature returned none
    best_feature = find_highest_gain_ratio_value(data, target_feature)
    if best_feature is None:
        return data[target_feature].mode()[0]

    best_feature_values = data[best_feature].unique()

    tree = {best_feature: {}}
    for best_feature_value in best_feature_values:
        subtree = build_tree(
            divide_data(data, best_feature)[best_feature_value],
            min_data,
            target_feature,
            current_depth=current_depth + 1,
            max_depth=max_depth
        )
        tree[best_feature][best_feature_value] = subtree

    return tree

def classify(tree, sample):
    """
    Classify a sample using the decision tree.

    Parameters
    ----------
    tree : dict
        Decision tree.
    sample : pandas.Series
        Input sample.

    Returns
    -------
    Any
        Predicted class label.
    """
    # if sample is a leaf, returns the value 1 or 0
    if not isinstance(tree, dict):
        return tree

    feature = next(iter(tree))
    value = sample[feature]
    # if value didn't appear in training data
    if value not in tree[feature]:
        return None
    subtree = tree[feature][value]
    return classify(subtree, sample)

def predict(tree, data):
    """
    Predict class labels for the given data using the decision tree.

    Parameters
    ----------
    tree : dict
        Decision tree.
    data : pandas.DataFrame
        Input data.

    Returns
    -------
    list
        List of predicted class labels.
    """
    test_res_predicted = []

    # function to classify single sample and append it to test_res_predicted
    def predict_row(row):
        test_res_predicted.append(classify(tree, row))

    # apply predict_row on all data
    data.apply(predict_row, axis=1)
    return test_res_predicted

def accuracy(predicted, result):
    """
    Calculate the accuracy of the predictions.

    Parameters
    ----------
    predicted : list
        List of predicted class labels.
    result : pandas.Series
        True class labels.

    Returns
    -------
    float
        Calculated accuracy.
    """
    count = 0
    for i in range(len(predicted)):
        if predicted[i] == result.values[i]:
            count += 1
    return count / len(predicted)

