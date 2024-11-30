import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from LogisticRegression import LogisticRegression as lr
from LightGBM import LightGBM as lgb
from RandomForest import RandomForest as rf  # Assuming RandomForest is implemented as above
from sklearn.model_selection import ParameterGrid

# Load the Iris dataset
def load_data():
    """Load and prepare the Iris dataset for classification."""
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target
    # Uncomment for binary classification
    mask = y <= 1
    X, y = X[mask], y[mask]
    
    return X, y


def evaluate(model, X_test, y_test):
    """Evaluate a model on the test set."""

    if model.__class__.__name__ == "LightGBM":
        if not isinstance(X_test, np.ndarray):
            X_test = np.array(X_test)
        if not isinstance(y_test, np.ndarray):
            y_test = np.array(y_test)
        
    y_pred = model.predict(X_test)

    return accuracy_score(y_test, y_pred)

def train_model(model, X_train, y_train):
    """Train a model on the training set."""
    if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
        print("Warning: Training data contains NaN or Inf values.")

    if model.__class__.__name__ == "RandomForest":
        print("Training Random Forest Model...")

        data_train = pd.concat([X_train, y_train], axis=1)
        data_train.columns = list(X_train.columns) + ["class"]        
        model.fit(data_train)

    elif model.__class__.__name__ == "LightGBM":
        print("Training LightGBM Model...")

        # Ensure X and y are numpy arrays
        if not isinstance(X_train, np.ndarray):
            X_train = np.array(X_train)
        if not isinstance(y_train, np.ndarray):
            y_train = np.array(y_train)

        model.fit(X_train, y_train)

    else:
        print("Training Logistic Regression Model...")
        model.fit(X_train, y_train)

def main():
    X, y = load_data()
    X = X.dropna().drop_duplicates().reset_index(drop=True)
    y = y.loc[X.index].reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


    # Train and evaluate models
    models = [
        (lr, {"learning_rate": [0.001, 0.01, 0.1], 'epsilon':[1e-6, 1e-5, 1e-4], "max_iter": [100, 200, 500]}),
        (rf, {"n_trees": [4,5,6], "max_depth": [3, 5, 7], "min_data": [5, 10, 15]}),
        (lgb, {"n_estimators": [10, 20, 50], "learning_rate": [0.001, 0.01, 0.1]}),
    ]
    for model_class, hyperparams in models:
        print("\n", model_class.__name__ + ":")
        model = gridsearch(X_train, y_train, model_class, hyperparams)
        evaluate(model, X_test, y_test)


def gridsearch(X_train, y_train, model_class, hyperparams):
    """Simple grid search over hyperparameters."""

    # Find the best combination of hyperparameters
    best_score = 0
    best_model = None
    best_params = None

    for i, hyperparam_dict in enumerate(ParameterGrid(hyperparams)):
        model = model_class(**hyperparam_dict)
        train_model(model, X_train, y_train)
        score = evaluate(model, X_train, y_train)

        print(f"{i+1:2d}/{len(list(ParameterGrid(hyperparams)))}: {hyperparam_dict} -> {score:.3f}")

        if score > best_score:
            best_score = score
            best_model = model
            best_params = hyperparam_dict

    print(f"Model = {model_class.__name__}, Best Parameters: {best_params} -> {best_score:.3f}")

    return best_model



if __name__ == "__main__":
    main()

