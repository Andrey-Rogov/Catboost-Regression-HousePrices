import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def garage(data: pd.DataFrame) -> pd.DataFrame:
    data['PrebGarage'] = data['GarageYrBlt'] == data['YearBuilt']
    data['PrebGarage'] = data['PrebGarage'].astype(np.int64)
    return data


# def bath(data: pd.DataFrame) -> pd.DataFrame:
#     mask1 = data['FullBath'] > 0
#     mask2 = data['HalfBath'] > 0
#     data['BothBaths'] = mask1 * mask2
#     data['BothBaths'] = data['BothBaths'].astype(np.int64)

#     mask1 = data['BsmtFullBath'] > 0
#     mask2 = data['BsmtHalfBath'] > 0
#     data['BothBathsBsm'] = mask1 * mask2
#     data['BothBathsBsm'] = data['BothBathsBsm'].astype(np.int64)
#     return data


class FeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.train_data = None
        self.train_target = None

    def fit(self, x_train, y_train):
        self.train_data = x_train
        self.train_target = y_train
        return self

    def transform(self, x_train):
        self.train_data = garage(x_train)
#         self.train_data = bath(x_train)
        return self.train_data
