import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

def load_and_split_data():
    """Generates and splits synthetic churn data."""
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=5, n_redundant=0, 
        n_classes=2, n_clusters_per_class=1, flip_y=0.1, random_state=42
    )
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    df['churn'] = y
    
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop('churn', axis=1), df['churn'], test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test
