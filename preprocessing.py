import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE


def handle_missing_values(dataframe, option='drop'):
    """
    Handle missing values in a DataFrame based on specified strategy.
    """
    if dataframe is None:
        raise ValueError('Dataframe argument cannot be None')

    missing_count = dataframe.isna().sum().sum()
    if missing_count == 0:
        print('No Missing Values Detected, returning original DataFrame.')
        return dataframe

    valid_options = ['remove', 'max', 'min', 'mean', 'median', 'mode']
    if option.lower() not in valid_options:
        raise ValueError(f'Option must be one of: {" | ".join(valid_options)}. Received: {option}')

    df = dataframe.copy()

    if option.lower() == 'drop':
        return df.dropna().reset_index(drop=True)
    elif option.lower() == 'mean':
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    elif option.lower() == 'median':
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    elif option.lower() == 'mode':
        for column in df.columns:
            df[column] = df[column].fillna(df[column].mode()[0])
    elif option.lower() == 'max':
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].max())
    elif option.lower() == 'min':
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].min())
    
    return df



def preprocessing_pipeline(dataframe, 
                           target_column, 
                           missing_values_option='drop',
                           random_state:int=42, 
                           verbose:bool=False
                           ):
    """
    Comprehensive data preprocessing pipeline with multiple advanced techniques.
    """
    data = dataframe.copy()
    if dataframe.duplicated().sum() > 0:
        if verbose: print('dropping duplicates ...', end='')
        data = data.drop_duplicates(inplace=False).reset_index(drop=True)
        if verbose: print('done')

    if dataframe.isna().sum().sum() > 0:
        if verbose: print('handling missing values ...', end='')
        data = handle_missing_values(data, option=missing_values_option)
        if verbose: print('done')


    if len(np.unique(dataframe[target_column])) > 1:
        smote = SMOTE(random_state=random_state)
        X_processed, y_processed = smote.fit_resample(
            data.drop(columns=[target_column]),
            data[target_column]
        )
        dataframe_processed = pd.concat(
            [pd.DataFrame(X_processed, columns=dataframe.drop(columns=[target_column]).columns),
            pd.Series(y_processed, name=target_column)],
            axis=1
        )

    return dataframe_processed
