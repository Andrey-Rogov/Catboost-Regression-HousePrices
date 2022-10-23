import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
TO_LE = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
     'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 
     'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
     'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir',
     'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 
     'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition', 'Alley',
     'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical',
     'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']


def fill_mean(data: pd.DataFrame, cols: list) -> pd.DataFrame:
    for col in cols:
        if col == 'GarageYrBlt':
            data[col].fillna(round(data[col].mean()), inplace=True)
            data[col] = data[col].astype(np.int64, copy=True)
        else:
            data[col].fillna(data[col].mean(), inplace=True)
    return data


def drop_unnecessary_cols(data: pd.DataFrame) -> pd.DataFrame:
    data.drop(['Id'], axis=1, inplace=True)
    return data


def labels_encoding(data: pd.DataFrame, cols: list) -> pd.DataFrame:
    for col in cols:
        enc = LabelEncoder()
        enc.fit(data[col].unique())
        a = enc.transform(data[col])
        data[col] = a
    return data


def prep_test(data: pd.DataFrame) -> pd.DataFrame:
    data = fill_mean(data, ['LotFrontage', 'MasVnrArea', 'GarageYrBlt', 'BsmtFinSF2', 
                                                'BsmtUnfSF', 'TotalBsmtSF','BsmtFullBath', 'BsmtHalfBath', 
                                                'GarageArea', 'GarageCars', 'BsmtFinSF1'])
    data = drop_unnecessary_cols(data)
    data = labels_encoding(data, TO_LE)
    return data


class PreprocessTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.train_data = None
        self.train_target = None

    def fit(self, x_train, y_train):
        self.train_data = x_train
        self.train_target = y_train
        return self

    def transform(self, x_train):
        self.train_data = fill_mean(x_train, ['LotFrontage', 'MasVnrArea', 'GarageYrBlt'])
        self.train_data = drop_unnecessary_cols(x_train)
        self.train_data = labels_encoding(x_train, TO_LE)
        return self.train_data
