# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 14:15:48 2022

@author: s7522
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
data = pd.read_csv("C:/Users/s7522/Downloads/3_Giris_Kavramlari_ve_Notasyon/3_Giris_Kavramlari_ve_Notasyon/1_Giris_Kavramlari/Wage.csv")
data.drop(["sex","region"], axis = 1, inplace = True)
print(data)
data.info()
print(data.isnull().any())
print(data.describe())

data_one_hot = pd.get_dummies(data, columns = ["maritl","race","jobclass","health_ins"])
#print(data_one_hot)
data_one_hot.drop(["health_ins_2. No","jobclass_2. Information"], inplace = True, axis= 1)

enc = OrdinalEncoder()
data_one_hot[["education","health"]] = enc.fit_transform(data_one_hot[["education","health"]])

independent = data_one_hot.iloc[:,[0,1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,17]]
dependent = data_one_hot.iloc[:,6]
X_train,X_test,Y_train,Y_test = train_test_split(independent,dependent,test_size=0.33, random_state=0)

X_train = X_train.iloc[:,[0,6]]
X_test = X_test.iloc[:,[0,6]]
linReg = LinearRegression()
linReg.fit(X_train,Y_train)

y_pred = linReg.predict(X_test)

test_list = data_one_hot.iloc[:,:].values
test_list = np.array(test_list, dtype=float)
wage = data.iloc[:,10]
model = sm.OLS(wage,test_list).fit()
print(model.summary())
#print(y_pred)