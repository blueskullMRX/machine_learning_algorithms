
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import cross_val_score, train_test_split
import numpy as np, pandas as pd


def backward_elimination_wrapper(estimator_cls, X, y, threshold=0.05, verbose=False, **estimator_params):
    """
    Perform Backward Elimination Wrapper Feature Selection with an estimator.
    
    Parameters:
    - estimator_cls: Class of the estimator to evaluate feature importance (e.g., RandomForest)
    - X: DataFrame of features
    - y: Series of target variable
    - threshold: p-value threshold for feature significance
    - verbose: Whether to print the progress and feature selection steps
    - **estimator_params: Additional parameters for the estimator
    
    Returns:
    - selected_features: List of selected feature indices
    - performance_scores: Performance scores at each step
    """
    # Copy of the original feature matrix
    X_copy = X.copy()
    
    # Get initial feature indices
    feature_indices = list(range(X.shape[1]))
    
    # Create the initial model instance and fit it
    if verbose : print("Starting Backward Elimination Wrapper...")
    estimator = estimator_cls(**estimator_params)
    estimator.fit(X_copy, y)
    baseline_score = estimator.score(X_copy, y)
    
    # Store performance scores at each step
    performance_scores = [baseline_score]
    
    if verbose:
        print(f"\t->Initial score: {baseline_score:.4f}")
    
    iteration = 0
    while len(feature_indices) > 1:
        # Compute feature importances using F-regression
        _, p_values = f_regression(X_copy, y)

        # Find the most insignificant feature
        max_p_index = np.argmax(p_values)
        if p_values[max_p_index] <= 0.05:
            if verbose: 
                print(f"\t--> All remaining features have p-value <= {threshold}. Stopping elimination process.")
            break
            
        # Remove the feature
        removed_index = feature_indices.pop(max_p_index)
        X_copy = X_copy.drop(columns=X_copy.columns[max_p_index])
        
        # Create a new instance of the estimator and refit it
        del estimator
        estimator = estimator_cls(**estimator_params)
        estimator.fit(X_copy, y)
        current_score = estimator.score(X_copy, y)
        
        # If performance significantly drops, add the feature back
        if current_score < baseline_score * (1 - threshold):
            feature_indices.append(removed_index)
            feature_indices.sort()
            X_copy = X.copy().iloc[:, feature_indices]  # Restore the dropped column
            if verbose:
                print(f"\t-->Restoring feature {removed_index} (p-value: {p_values[max_p_index]:.4f}) due to performance drop.")
            break # because we already tried the least important feature, no reason to continue
        else:
            baseline_score = current_score
            performance_scores.append(current_score)
            
            if verbose:
                print(f"\t-->Iteration {iteration + 1}: Removed feature {removed_index} (p-value: {p_values[max_p_index]:.4f}), "
                      f"new score: {current_score:.4f}")
        
        iteration += 1

    return feature_indices, performance_scores


# Example usage
if __name__ == "__main__":
    from sklearn.preprocessing import StandardScaler

    # Step 1: Generate synthetic regression data
    X, y = make_classification(n_samples=10000, n_features=200, n_informative=20, random_state=42)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    
    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
    
    # Step 2: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: Initialize the estimator (e.g., Linear Regression)
    from models.RF import RandomForest
    from models.LR import LogisticRegression
    from sklearn.linear_model import LogisticRegression as FastLR
    from sklearn.tree import DecisionTreeClassifier
    from time import time
    
    startTime = time()

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    print('before backward elimination', model.score(X_test, y_test))


    # Step 4: Apply the backward elimination wrapper
    selected_features, performance_scores = backward_elimination_wrapper(DecisionTreeClassifier, X_train, y_train, threshold=0.05, verbose=True)

    # Step 5: Display the results
    print("\nSelected features indices:", selected_features)
    print("\nPerformance scores at each step:", performance_scores)

    # Step 6: Evaluate the final model with selected features on the test set
    X_train_selected = X_train.loc[:, X_train.columns[selected_features]]
    X_test_selected = X_test.loc[:, X_test.columns[selected_features]]

    model = DecisionTreeClassifier()
    # Fit the model on the selected features
    model.fit(X_train_selected, y_train)

    # Evaluate the model's performance on the test set
    test_score = model.score(X_test_selected, y_test)
    print(f"\nScore with selected features: {test_score:.4f}")

    print('End time : ', time() - startTime)
