import numpy as np
from sklearn.datasets import make_classification, make_blobs

def generate_classification_data(n_samples=100, n_features=2, n_classes=2, random_state=None):
    """
    Generate synthetic classification data.
    
    Parameters:
    - n_samples: int, number of samples
    - n_features: int, number of features
    - n_classes: int, number of classes
    - random_state: int, random seed for reproducibility
    
    Returns:
    - X: ndarray, feature matrix
    - y: ndarray, target labels
    """
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes, random_state=random_state)
    return X, y

def generate_cluster_data(n_samples=100, n_features=2, n_clusters=3, random_state=None):
    """
    Generate synthetic clustering data.
    
    Parameters:
    - n_samples: int, number of samples
    - n_features: int, number of features
    - n_clusters: int, number of clusters
    - random_state: int, random seed for reproducibility
    
    Returns:
    - X: ndarray, feature matrix
    - y: ndarray, cluster labels
    """
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=random_state)
    return X, y
