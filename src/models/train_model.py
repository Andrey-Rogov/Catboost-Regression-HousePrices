import pandas as pd
from catboost import CatBoostRegressor


def train(X_train: pd.DataFrame, y_train: pd.DataFrame) -> CatBoostRegressor:
    cat_model = CatBoostRegressor(n_estimators=100,
                                   depth=2,
                                   learning_rate=1, 
                                   loss_function='MAE',
                                 verbose=False)
    cat_model.fit(X_train, y_train)
    return cat_model
