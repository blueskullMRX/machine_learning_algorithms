import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.feature_selection import SelectFromModel

def ridge_feature_selection(X, y, alphas=None, cv=5, threshold='mean'):
    """
    Perform feature selection using Ridge Regression with cross-validated regularization.
    
    Parameters:
    -----------
    X : pandas DataFrame or numpy array
        Input features matrix
    y : pandas Series or numpy array
        Target variable
    alphas : array-like, optional (default=None)
        Array of alpha values to try for cross-validation
        If None, a default range of alphas will be used
    cv : int, optional (default=5)
        Number of folds for cross-validation
    threshold : str or float, optional (default='mean')
        Threshold for feature selection:
        - 'mean': select features with importance above mean
        - 'median': select features with importance above median
        - float: custom threshold value
    
    Returns:
    --------
    dict containing:
    - selected_features: list of selected feature names
    - feature_importances: pandas Series of feature importances
    - best_alpha: best regularization strength found by cross-validation
    - selector: fitted SelectFromModel object for further use
    """
    # Ensure input is numpy array
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns
        X = X.values
    else:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Prepare alphas if not provided
    if alphas is None:
        alphas = np.logspace(-3, 3, 100)
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform Ridge Regression with cross-validation
    ridge_cv = RidgeCV(alphas=alphas, cv=cv, scoring='neg_mean_squared_error')
    ridge_cv.fit(X_scaled, y)
    
    # Get the best alpha
    best_alpha = ridge_cv.alpha_
    
    # Create selector based on feature importances
    selector = SelectFromModel(ridge_cv, prefit=True, threshold=threshold)
    
    # Get feature importances (absolute values of coefficients)
    feature_importances = pd.Series(
        np.abs(ridge_cv.coef_), 
        index=feature_names
    ).sort_values(ascending=False)
    
    # Select features
    selected_mask = selector.get_support()
    selected_features = [
        feature_names[i] for i in range(len(feature_names)) 
        if selected_mask[i]
    ]
    
    return {
        'selected_features': selected_features,
        'feature_importances': feature_importances,
        'best_alpha': best_alpha,
        'selector': selector
    }

# Example usage
def example_usage():
    """
    Demonstrates how to use the ridge_feature_selection function
    """
    from sklearn.datasets import make_regression
    
    # Generate a sample regression dataset
    X, y = make_regression(n_samples=100, n_features=20, n_informative=10, noise=0.1, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    
    # Perform feature selection
    selection_results = ridge_feature_selection(X_df, y)
    
    print("Best Regularization Strength (Alpha):", selection_results['best_alpha'])
    print("\nFeature Importances:")
    print(selection_results['feature_importances'])
    
    print("\nSelected Features:")
    print(selection_results['selected_features'])

# Uncomment to run the example
if __name__ == '__main__':
    example_usage()