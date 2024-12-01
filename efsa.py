# machine_learning_project/efsa.py

import os, pandas as pd
from feature_selection import (
    corr_filter, 
    corr_filter_between,
    backward_selection_rf, 
    backward_selection_lr,
    recursive_feature_elimination_rf, 
    recursive_feature_elimination_lr,
)

def efsa_pipeline(dataframe, target_column, verbose=False, max_columns:int=25, min_cor:float=0.24):
    """
    Perform Enhanced Feature Selection Algorithm (EFSA).
    """
    from DecisionTreeC45 import detect_continuous_columns, to_category

    if not os.path.exists('to_category.csv'):
        if verbose: print('Transforming data into numerical and saving...', end='')
        data = dataframe.copy()

        continuous_columns = detect_continuous_columns(dataframe, threshold=10)
        for feature in continuous_columns:
            data[feature] = to_category(data, feature, target_column)
        data.to_csv('to_category.csv', index=False)
        if verbose: print('Done')
    else:
        if verbose: print('Data Found, importing...')
        data = pd.read_csv('to_category.csv')

    subsets = {}

    if verbose: print('Performing EFSA:')


    if verbose: print('\t--->Filter based wrapper... ', end='')
    selected_features = corr_filter_between(data, target_column, min_cor)
    subsets['Corr-Filter'] = selected_features
    if verbose: print('Done')


    if verbose: print("\t--->Backward elimination based wrapper: ", end='')
    selected_features = backward_selection_rf(data, target='class', max_columns=max_columns)
    subsets['RF-Wrapper'] = selected_features

    selected_features = backward_selection_lr(data, target='class',max_columns=max_columns)
    subsets['LR-Wrapper'] = selected_features
    
    # selected_features = backward_selection_lr(data, target='class', max_columns=max_columns)
    subsets['LGBM-Wrapper'] = selected_features
    if verbose: print('Done')


    if verbose: print('\t--->Recusrive Feature Elimination based Embedded: ', end='')
    selected_features = recursive_feature_elimination_rf(data, target='class', min_feature=max_columns)
    subsets['RF-Embedded'] = selected_features

    selected_features = recursive_feature_elimination_lr(data, target='class', min_feature=max_columns)
    subsets['LR-Embedded'] = selected_features

    # selected_features = recursive_feature_elimination_lr(data, target='class', min_feature=max_columns)
    subsets['LGBM-Embedded'] = selected_features
    if verbose: print('Done\n')


    if verbose: print(selected_features)
    print('\tEFSA done')

    return subsets

