import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from LogisticRegression import LogisticRegression as lr
from models.RF import RandomForest as rf 
from LightGBM import LightGBM as lgb
# from RandomForest import RandomForest as rf  
from sklearn.model_selection import ParameterGrid
from time import time

best_score = 0
best_params = {}
best_model = {}
best_models = {}

def load_data():
    """Load and prepare the Iris dataset for classification."""
    iris = load_breast_cancer(as_frame=True)
    X, y = iris.data, iris.target

    X = X.dropna().drop_duplicates()
    y = y.loc[X.index].reset_index(drop=True)

    return X, y

X, y = load_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Logistic Regression Model
param_grid_lr = {
    'learning_rate': [0.0001, 0.001, 0.01, 0.1],   
    'epsilon': [1e-4, 1e-3, 1e-2, 1e-1],
    'max_iter': [10, 100, 1000, 10000],
}

total_iterations = len(ParameterGrid(param_grid_lr))
print(f'Testing {lr.__name__}')
for i, hyperparam_dict in enumerate(ParameterGrid(param_grid_lr), start=1):
    model = lr(**hyperparam_dict)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)

    print(f"\t {i}/{total_iterations} -> : {hyperparam_dict} | Accuracy: {score:.3f}")

    if score > best_score:
        best_score = score
        best_model = model
        best_params = hyperparam_dict

best_models[best_model] = best_params

param_grid_rf = {
    'n_estimators': [5, 10, 15, 20],
    'max_depth': [None, 3, 5, 7, 9, 11],
    'max_features': [None],
}

# Training the Random Forest Model
total_iterations = len(ParameterGrid(param_grid_rf))
print(f'Testing {rf.__name__}')
for i, hyperparam_dict in enumerate(ParameterGrid(param_grid_rf), start=1):
    model = rf(**hyperparam_dict)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)

    print(f"\t{i}/{total_iterations} -> : {hyperparam_dict} | Accuracy: {score:.3f}")

    if score > best_score:
        best_score = score
        best_model = model
        best_params = hyperparam_dict

best_models[best_model] = best_params



print(f"""Best models and their hyperparameters:
{best_models} -> Accuracy: {best_score}
""")
