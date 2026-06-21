from sklearn.model_selection import train_test_split, cross_val_score
import joblib

from data_loading import load_data
from preprocessing import get_preprocessor
from model_training import get_models
from evaluation import evaluate_model
from utils import select_best_model


# =========================
# 1. LOAD DATA
# =========================
X, y = load_data()


# =========================
# 2. TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================
# 3. PREPROCESSOR
# =========================
preprocessor = get_preprocessor(X_train)


# =========================
# 4. MODELS
# =========================
models = get_models(preprocessor)


# =========================
# 5. CROSS VALIDATION
# =========================
cv_results = {}

print("\nCROSS VALIDATION RESULTS")
print("-" * 30)

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    cv_results[name] = scores.mean()
    print(name, ":", scores.mean())


# =========================
# 6. BEST MODEL SELECTION
# =========================
best_model_name = select_best_model(cv_results)
best_model = models[best_model_name]

print("\nBest Model:", best_model_name)


# =========================
# 7. TRAIN BEST MODEL
# =========================
best_model.fit(X_train, y_train)


# =========================
# 8. EVALUATION
# =========================
evaluate_model(best_model_name, best_model, X_test, y_test)


# =========================
# 9. SAVE MODEL
# =========================
joblib.dump(best_model, "model.pkl")
print("\nModel saved as model.pkl")