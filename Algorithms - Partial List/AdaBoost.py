from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import os
import random as rn

def AdaBoost_FS(X_train, y_train, k, row, random_state):
    """
    Select features using the AdaBoost classifier with a Decision Tree base estimator. This function returns the indices
    of the top 'k' important features based on their importance scores, along with their normalized importance values and
    the accumulated importance.

    Args:
        X_train (numpy array): Training data with features.
        y_train (numpy array): Labels corresponding to the training data.
        k (int): Number of top features to select.
        row (dict): Dictionary containing hyperparameters, including:
            max_depth (int): The maximum depth of the base decision tree estimators.
            n_estimators (int): The number of boosting stages to perform.
            learning_rate (float): Weight applied to each classifier at each boosting iteration.
        random_state (int): Seed used by the random number generator for reproducibility.

    Returns:
        tuple: (indices, normalized_importances, accumulated_importance)
            - indices (list): Indices of the top 'k' important features.
            - normalized_importances (list): Normalized importance scores of the selected features.
            - accumulated_importance (float): Sum of the normalized importance scores.
    """
    
    # Set the seed for reproducibility
    seed = random_state
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    rn.seed(seed)

    # Check if the values are continuous
    if np.issubdtype(X_train.dtype, np.number):
        algorithm = 'SAMME.R'
    else:
        algorithm = 'SAMME'
    
    # Initialize the classifier with hyperparameters
    abc = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=row['max_depth']),
        n_estimators=row['n_estimators'],
        learning_rate=float(row['learning_rate']),
        algorithm=algorithm,  
        random_state=random_state
    )

    
    # Fit the classifier to the training data
    abc.fit(X_train, y_train)

    # Get feature importances from the fitted classifier
    feature_importances = abc.feature_importances_

    # Get the indices of the top 'k' features
    indices = np.argsort(feature_importances)[-k:]

    return indices






    """
    Comment: Another option is to use the feature importance from the AdaBoost classifier to select the top 'k' features. For standardization, we use the Etree for all algorithms.

    
    
    # Get feature importances
    importances = abc.feature_importances_
    
    # Normalize importances based on all features
    normalized_importances = importances / np.sum(importances)

    # Get indices of the top 'k' important features in descending order based on normalized importances
    indices = np.argsort(-normalized_importances)[:k]

    # Accumulated importance of the normalized importances
    accumulated_importance = np.round(np.sum(normalized_importances[indices]), 3)
    if accumulated_importance > 1:  # Correct if sum exceeds 1 due to rounding
        accumulated_importance = 1

    # Extract the normalized scores for the top 'k' features for output
    top_k_normalized_importances = normalized_importances[indices]

    return list(indices), list(np.round(top_k_normalized_importances, 3)), accumulated_importance
    """