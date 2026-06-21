from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def get_models(preprocessor):
    models = {
        "Linear Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LinearRegression())
        ]),

        "Decision Tree": Pipeline([
            ("preprocessor", preprocessor),
            ("model", DecisionTreeRegressor(random_state=42))
        ]),

        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(random_state=42))
        ])
    }

    return models