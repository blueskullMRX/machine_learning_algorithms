import numpy as np
from RandomForest import train_random_forest, predict_random_forest, accuracy, gain_ratio
import pandas as pd
from models.LR import train_lr, predict
from sklearn.preprocessing import StandardScaler

# Feature selection methods

# Feature selection based on Filter method
def corr_filter(df, target, min_cor=0.35):
    """
    Filter features based on correlation with target.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    target : str
        Target feature.
    min_cor : float, optional
        Minimum absolute correlation, by default 0.35.

    Returns
    -------
    list
        List of features that have correlation with target greater than or equal to min_cor.
    """
    corr = df.corr()[target].abs()
    return corr[corr >= min_cor].index.difference([target]).tolist()

def corr_filter_between(df, target, min_cor=0.35):
    """
    Filter features based on correlation with target and between selected features.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    target : str
        Target feature.
    min_cor : float, optional
        Minimum absolute correlation, by default 0.35.

    Returns
    -------
    list
        List of features that have correlation with target greater than or equal to min_cor 
        and are not highly correlated with each other.
    """
    corr = df.corr()
    target_corr = corr[target].abs()
    selected_features = target_corr[target_corr >= min_cor].index.difference([target])
    # Find highly correlated features among selected features
    corr_between = corr.loc[selected_features, selected_features]
    highly_correlated = (corr_between >= 1 - min_cor).any(axis=0)
    # Remove highly correlated features from the selected list
    return selected_features.difference(corr_between.columns[highly_correlated]).tolist()

# Feature selection based on Wrapper method
def to_accuracy_rf(data, target):
    """
    Calculate accuracy of Random Forest on the given data.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataframe.
    target : str
        Target feature.

    Returns
    -------
    float
        Accuracy of the Random Forest model.
    """
    trees, _ = train_random_forest(data, tree_max_depth=8, nbr_trees=5)
    pred = predict_random_forest(trees, data)
    return accuracy(pred, data[target])

def to_accuracy_lr(data, target):
    """
    Calculate accuracy of Logistic Regression on the given data.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataframe.
    target : str
        Target feature.

    Returns
    -------
    float
        Accuracy of the Logistic Regression model.
    """
    x = data.drop(target, axis='columns').values
    y = data[target].values
    y_df = data[target]
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    w, b = train_lr(x, y, epsilon=0.00001, max_iteration=3000)
    pred = predict(x, w, b)
    return accuracy(pred, y_df)

def backward_selection_rf(data, target):
    """
    Perform backward feature selection using Random Forest.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataframe.
    target : str
        Target feature.

    Returns
    -------
    list
        List of selected features.
    """
    features = data.columns.to_list()
    remaining_features = features
    acc = to_accuracy_rf(data, target)
    for feature in features:
        if feature == target:
            continue
        temp = remaining_features
        temp.remove(feature)
        temp_acc = to_accuracy_rf(data[temp], target)
        if temp_acc >= acc:
            acc = temp_acc
            remaining_features = temp
    remaining_features.remove(target)
    return remaining_features

def backward_selection_lr(data, target):
    """
    Perform backward feature selection using Logistic Regression.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataframe.
    target : str
        Target feature.

    Returns
    -------
    list
        List of selected features.
    """
    features = data.columns.to_list()
    remaining_features = features
    acc = to_accuracy_lr(data, target)
    for feature in features:
        if feature == target:
            continue
        temp = remaining_features
        temp.remove(feature)
        temp_acc = to_accuracy_lr(data[temp], target)
        if temp_acc >= acc:
            acc = temp_acc
            remaining_features = temp
    remaining_features.remove(target)
    return remaining_features

# Feature selection based on Embedded method
def recursive_feature_elimination_rf(train_data, target, min_feature=1):
    """
    Perform recursive feature elimination using Random Forest.

    Parameters
    ----------
    train_data : pandas.DataFrame
        Input dataframe.
    target : str
        Target feature.
    min_feature : int, optional
        Minimum number of features to select, by default 1.

    Returns
    -------
    list
        List of selected features.
    """
    data = train_data.copy()
    features = data.columns.to_list()
    features.remove(target)

    importances = {}
    for feature in features:
        importances[feature] = gain_ratio(data, feature, target)

    # Eliminate least important features until reaching min_feature
    while len(features) > min_feature:
        data = data[features]
        data[target] = train_data[target]
        
        least_important_feature = min(importances, key=importances.get)
        
        features.remove(least_important_feature)
        del importances[least_important_feature]

    return features

def recursive_feature_elimination_lr(train_data, target, min_feature=1):
    """
    Perform recursive feature elimination using Logistic Regression.

    Parameters
    ----------
    train_data : pandas.DataFrame
        Input dataframe.
    target : str
        Target feature.
    min_feature : int, optional
        Minimum number of features to select, by default 1.

    Returns
    -------
    list
        List of selected features.
    """
    data = train_data.copy()
    features = data.columns.to_list()
    features.remove(target)

    train_x = train_data[features].values
    train_y = train_data[target].values
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)

    importances = {}
    for i, feature in enumerate(features):
        w, _ = train_lr(train_x, train_y, epsilon=0.00001, max_iteration=3000)
        importances[feature] = np.abs(w[i])

    # Eliminate least important features until reaching min_feature
    while len(features) > min_feature:
        data = data[features]
        data[target] = train_data[target]
        
        least_important_feature = max(importances, key=importances.get)
        
        features.remove(least_important_feature)
        del importances[least_important_feature]

    return features
