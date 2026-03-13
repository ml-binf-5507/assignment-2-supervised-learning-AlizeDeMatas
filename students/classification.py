"""
Classification functions for logistic regression and k-nearest neighbors.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


def train_logistic_regression_grid(X_train, y_train, param_grid=None):
    """
    Train logistic regression models with grid search over hyperparameters.
    """
    if param_grid is None:
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['lbfgs']
        }

    # Create LogisticRegression with max_iter=1000
    log_reg = LogisticRegression(max_iter=1000)

    # Use GridSearchCV with cv=5
    grid_search = GridSearchCV(
        estimator=log_reg,
        param_grid=param_grid,
        cv=5
    )

    # Fit on training data
    grid_search.fit(X_train, y_train)

    # Return fitted GridSearchCV object
    return grid_search



def train_knn_grid(X_train, y_train, param_grid=None):
    """
    Train k-NN models with grid search over hyperparameters.
    
    Parameters
    ----------
    X_train : np.ndarray or pd.DataFrame
        Training feature matrix (should be scaled)
    y_train : np.ndarray or pd.Series
        Training target vector (binary)
    param_grid : dict, optional
        Parameter grid for GridSearchCV.
        Default: {'n_neighbors': [3, 5, 7, 9, 11, 15, 20],
                  'weights': ['uniform', 'distance'],
                  'metric': ['euclidean', 'manhattan']}
        
    Returns
    -------
    sklearn.model_selection.GridSearchCV
        Fitted GridSearchCV object with best model
    """
    if param_grid is None:
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 15, 20],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
    knn = KNeighborsClassifier()

    grid_search = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        scoring="roc_auc",  # AUROC for binary classification
        cv=5,
        n_jobs=-1,
        return_train_score=False,
    )

    grid_search.fit(X_train, y_train)

    return grid_search
    


def get_best_logistic_regression(X_train, y_train, X_test, y_test, param_grid=None):
    """
    Get best logistic regression model with test R² evaluation.
    
    Parameters
    ----------
    X_train : np.ndarray or pd.DataFrame
        Training features
    y_train : np.ndarray or pd.Series
        Training target
    X_test : np.ndarray or pd.DataFrame
        Test features
    y_test : np.ndarray or pd.Series
        Test target
    param_grid : dict, optional
        Parameter grid for GridSearchCV
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'model': best fitted LogisticRegression model
        - 'best_params': best parameters found
        - 'cv_results_df': DataFrame of all CV results
    """
    # Run grid search on training data
    grid_search = train_logistic_regression_grid(X_train, y_train, param_grid=param_grid)

    # Extract best model and params
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Put CV results in a DataFrame for inspection / plotting
    cv_results_df = pd.DataFrame(grid_search.cv_results_)

    return {
        "model": best_model,
        "best_params": best_params,
        "cv_results_df": cv_results_df,
    }
 


def get_best_knn(X_train, y_train, X_test, y_test, param_grid=None):
    """
    Get best k-NN model with test R² evaluation.
    
    Parameters
    ----------
    X_train : np.ndarray or pd.DataFrame
        Training features (scaled)
    y_train : np.ndarray or pd.Series
        Training target
    X_test : np.ndarray or pd.DataFrame
        Test features (scaled)
    y_test : np.ndarray or pd.Series
        Test target
    param_grid : dict, optional
        Parameter grid for GridSearchCV
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'model': best fitted KNeighborsClassifier model
        - 'best_params': best parameters found
        - 'best_k': best n_neighbors value
        - 'cv_results_df': DataFrame of all CV results
    """
    # Run grid search on training data
    grid_search = train_knn_grid(X_train, y_train, param_grid=param_grid)

    # Extract best model and params
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Extract best k directly from the best_params
    best_k = best_params.get("n_neighbors", None)

    # Wrap cv_results_ in a DataFrame for convenience
    cv_results_df = pd.DataFrame(grid_search.cv_results_)

    return {
        "model": best_model,
        "best_params": best_params,
        "best_k": best_k,
        "cv_results_df": cv_results_df,
    }