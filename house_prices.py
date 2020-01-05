# -*- coding: utf-8 -*-
#kaggle house-prices-advanced-regression-techniques
from scipy import *
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import warnings
import time

warnings.filterwarnings('ignore')

STARTTIME = time.time()

directory = "/Users/yumi/Documents/house-prices-advanced-regression-techniques/"

TRAIN = pd.read_csv(directory+'train.csv')
TEST = pd.read_csv(directory+'test.csv')

X = TRAIN.copy()
y = TRAIN['SalePrice']
X.drop('Id',axis=1,inplace=True)
X.drop('SalePrice',axis=1,inplace=True)
print("shape(X): {}, shape(y): {}".format(shape(X),shape(y)))

dtype_list = array([X[x].dtype for x in X.columns])
want_1 = dtype_list == "float64"
want_2 = dtype_list == "int64"
cols = X.columns[want_1 + want_2]

X = X[cols]
#X.dropna(axis = 1, inplace = True)
X.fillna(value = 0, inplace = True) #no dropping allowed!
print("shape(X): {}, shape(y): {}".format(shape(X),shape(y)))

X_train, X_test, y_train, y_test = train_test_split(X, y)

logreg = LinearRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print("Log Reg Accuracy: {}".format(logreg.score(X_test, y_test)))

print("Time elapsed; {:.2f}s".format(time.time()-STARTTIME))

RRR = RandomForestRegressor()
RRR.fit(X_train, y_train)
y_pred = RRR.predict(X_test)
print("Random Forest Accuracy: {}".format(RRR.score(X_test, y_test)))

coeff = logreg.coef_ / max(abs(logreg.coef_))
feat_imp = RRR.feature_importances_ / max(abs(RRR.feature_importances_))
plt.plot(coeff,"-ko")
plt.plot(feat_imp,"-r*")
plt.legend(["LogReg","RdmFst"])
plt.grid("on")
plt.show()

print("Time elapsed; {:.2f}s".format(time.time()-STARTTIME))