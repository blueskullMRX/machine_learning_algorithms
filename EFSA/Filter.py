import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, List, Optional

class PearsonFeatureSelector:
    def __init__(
        self, 
        correlation_threshold: float = 0.7, 
        method: str = 'pearson',
        absolute: bool = False
    ):
        """
        Initialize Pearson Feature Selector

        Parameters:
        -----------
        correlation_threshold : float, default=0.7
            Threshold for correlation between features
        method : str, default='pearson'
            Correlation method. Options: 'pearson', 'spearman', 'kendall'
        absolute : bool, default=False
            Whether to use absolute correlation values
        """
        self.correlation_threshold = correlation_threshold
        self.method = method
        self.absolute = absolute
        
        # Attributes to store results
        self.correlation_matrix_ = None
        self.selected_features_ = None
        self.feature_importances_ = None

    def fit(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ):
        """
        Compute feature correlations and select relevant features

        Parameters:
        -----------
        X : DataFrame or numpy array
            Input features
        y : Series or numpy array, optional
            Target variable (for supervised feature selection)

        Returns:
        --------
        self : PearsonFeatureSelector
            Fitted feature selector
        """
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        # Compute correlation matrix
        if y is not None:
            temp_df = X.copy()
            temp_df['target'] = y
            corr_matrix = temp_df.corr(method=self.method)
            
            # Get feature correlations with target
            target_correlations = corr_matrix['target'][:-1]
            
            # Compute feature-to-feature correlation matrix
            feature_corr_matrix = X.corr(method=self.method)
        else:
            # Compute correlation matrix for unsupervised selection
            corr_matrix = X.corr(method=self.method)
            feature_corr_matrix = corr_matrix
            target_correlations = None
        
        # Store correlation matrix
        self.correlation_matrix_ = feature_corr_matrix
        
        # Use absolute correlation if specified
        if self.absolute:
            feature_corr_matrix = np.abs(feature_corr_matrix)
            if target_correlations is not None:
                target_correlations = np.abs(target_correlations)
        
        # Select features
        selected_features = self._select_features(
            feature_corr_matrix, 
            target_correlations
        )
        
        self.selected_features_ = selected_features
        
        # Compute feature importances (correlation strength)
        if target_correlations is not None:
            self.feature_importances_ = np.abs(target_correlations[selected_features])
        
        return self

    def _select_features(
        self, 
        feature_corr_matrix: pd.DataFrame, 
        target_correlations: Optional[pd.Series] = None
    ) -> List[str]:
        """
        Select features based on correlation matrix

        Parameters:
        -----------
        feature_corr_matrix : DataFrame
            Correlation matrix between features
        target_correlations : Series, optional
            Correlations with target variable

        Returns:
        --------
        List of selected feature names
        """
        # If target correlations provided, prioritize features correlated with target
        if target_correlations is not None:
            # Sort features by absolute correlation with target
            sorted_target_corr = target_correlations.abs().sort_values(ascending=False)
            feature_order = sorted_target_corr.index.tolist()
        else:
            # If no target, use feature names as is
            feature_order = feature_corr_matrix.columns.tolist()
        
        selected_features = []
        for feature in feature_order:
            # Skip if feature already selected
            if feature in selected_features:
                continue
            
            # Add current feature
            selected_features.append(feature)
            
            # Check correlations with already selected features
            for selected in selected_features[:-1]:
                corr_value = feature_corr_matrix.loc[feature, selected]
                
                # Remove highly correlated features
                if abs(corr_value) >= self.correlation_threshold:
                    selected_features.remove(feature)
                    break
        
        return selected_features

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Transform input data to selected features

        Parameters:
        -----------
        X : DataFrame or numpy array
            Input features

        Returns:
        --------
        DataFrame with selected features
        """
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        # Select features
        if self.selected_features_ is None:
            raise ValueError("Must call fit() before transform()")
        
        return X[self.selected_features_]

    def plot_correlation_heatmap(
        self, 
        figsize: tuple = (12, 10), 
        cmap: str = 'coolwarm'
    ):
        """
        Plot correlation heatmap of features

        Parameters:
        -----------
        figsize : tuple, default=(12, 10)
            Size of the figure
        cmap : str, default='coolwarm'
            Colormap for the heatmap
        """
        if self.correlation_matrix_ is None:
            raise ValueError("Must call fit() before plotting")
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            self.correlation_matrix_, 
            annot=True, 
            cmap=cmap, 
            center=0, 
            vmin=-1, 
            vmax=1,
            square=True
        )
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.show()

# Example usage function
def example_usage():
    """
    Demonstrate Pearson Feature Selector functionality
    """
    # Generate synthetic dataset
    from sklearn.datasets import make_regression
    
    # Create dataset with some correlated features
    X, y = make_regression(
        n_samples=1000, 
        n_features=20, 
        n_informative=10, 
        noise=0.1, 
        random_state=42
    )
    
    # Convert to DataFrame for demonstration
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Initialize and fit Pearson Feature Selector
    selector = PearsonFeatureSelector(correlation_threshold=0.8, method='pearson', absolute=True)
    selector.fit(X_df, y)
    
    # Print selected features
    print("Selected features based on correlation with target variable:")
    print(selector.selected_features_)
    
    # # Plot the correlation heatmap
    # selector.plot_correlation_heatmap()
    
    # Transform data to selected features
    X_selected = selector.transform(X_df)
    print("Transformed data with selected features:")
    print(X_selected.head())


if __name__ == "__main__":
    # Run the example usage to see it in action
    example_usage()
