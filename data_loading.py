from sklearn.datasets import fetch_california_housing
import pandas as pd

def load_data():
    data = fetch_california_housing()
    
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["price"] = data.target
    
    X = df.drop("price", axis=1)
    y = df["price"]
    
    return X, y