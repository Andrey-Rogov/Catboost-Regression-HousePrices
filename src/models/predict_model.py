import pandas as pd
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor

def predict(model: CatBoostRegressor,
            X_train: pd.DataFrame,
            X_test: pd.DataFrame,
            y_train: pd.DataFrame,
            y_test: pd.DataFrame) -> list:
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    return r2_score(y_test, y_pred_test), r2_score(y_train, y_pred_train)