#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 15:41:13 2017

@author: akshay
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing data
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:,1].values
                
#splitting data into training and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)


#Fitting simplelinear regression model to our data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


# Predict the values
y_pred = regressor.predict(x_test)

#plot graph for training set
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'green')
plt.title('Salary vs Experience')
plt.xlabel('Years pf experience')
plt.ylabel('Salary')
plt.show()

#plot graph for the test set
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'green')
plt.title('Salary vs Experience')
plt.xlabel('Years pf experience')
plt.ylabel('Salary')
plt.show()