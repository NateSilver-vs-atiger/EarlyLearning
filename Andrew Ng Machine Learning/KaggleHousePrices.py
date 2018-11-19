
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pandas_profiling as ppd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler)

os.chdir("C://Users/Stacy/Documents/Eric's Courses/KaggleHousePrices/")

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

#pfr = ppd.ProfileReport(train_data)
#pfr.to_file(outputfile = 'HousingTrainData.html')
##Columns with missing data
categorical = ['Alley', 'BsmtCond', 'BsmtExposure', 'BsmtExposure', 'BsmtFinType1',  'BsmtFinType2', 'BsmtQual', 'Electrical', 'Fence', 'FireplaceQu', 'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType', 'MiscFeature', 'PoolQC']
numeric_skew = ['GarageYrBlt', 'LotFrontage']

cat_imputer = SimpleImputer(missing_values = np.nan, strategy= 'constant', fill_value= 'Not Specified')
num_imputer = SimpleImputer(missing_values = np.nan, strategy= 'median')

imputed_train = train_data
imputed_train[categorical] = cat_imputer.fit_transform(imputed_train[categorical])
imputed_train[numeric_skew] = num_imputer.fit_transform(imputed_train[numeric_skew])

imputed_test = test_data
imputed_test[categorical] = cat_imputer.transform(imputed_test[categorical])
imputed_test[numeric_skew] = num_imputer.transform(imputed_test[numeric_skew])

#Numeric Columns for standardizing
numeric_min_max = ['2ndFlrSF', '3SsnPorch', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'EnclosedPorch', 'LowQualFinSF', 'MasVnrArea', 'MiscVal', 'OpenPorchSF', 'PoolArea', 'ScreenPorch' , 'WoodDeckSF']
numeric_standard_scaler = ['1stFlrSF', 'BedroomAbvGr', 'GarageArea', 'GarageYrBlt', 'GrLivArea', 'LotFrontage', 'MSSubClass', 'MoSold', 'OverallCond', 'OverallQual', 'TotRmsAbvGrd', 'TotalBsmtSF', 'YearBuilt', 'YearRemodAdd', 'YrSold']
numeric_robust_scaler = ['LotArea']
##For now, chose not to normalize SalePrice
##Turn 'Id' into an index or drop it entirely
standardscale = StandardScaler()
minmaxscale = MinMaxScaler()
robustscale = RobustScaler()

scaled_train = imputed_train
scaled_train[numeric_min_max] = minmaxscale.fit_transform(scaled_train[numeric_min_max])
scaled_train[numeric_standard_scaler] = standardscale.fit_transform(scaled_train[numeric_standard_scaler])
scaled_train[numeric_robust_scaler] = robustscale.fit_transform(scaled_train[numeric_robust_scaler])

#pfr = ppd.ProfileReport(scaled_train)
#pfr.to_file(outputfile = 'ScaledandImputedHousingData.html')

scaled_test = imputed_test
scaled_test[numeric_min_max] = minmaxscale.transform(scaled_test[numeric_min_max])
scaled_test[numeric_standard_scaler] = standardscale.transform(scaled_test[numeric_standard_scaler])
scaled_test[numeric_robust_scaler] = robustscale.transform(scaled_test[numeric_robust_scaler])

dummies_train = scaled_train
dummies_train = pd.get_dummies(dummies_train)

dummies_test = scaled_test
dummies_test = pd.get_dummies(dummies_test)