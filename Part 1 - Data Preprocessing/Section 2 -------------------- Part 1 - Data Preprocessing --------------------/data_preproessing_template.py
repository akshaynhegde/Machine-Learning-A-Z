# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot

#importing data
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:,3].values

#splitting data into training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

##feature scaling
#from sklearn.preprocessing import StandardScaler
#scaler_x = StandardScaler()
#x_train = scaler_x.fit_transform(x_train)
#x_test = scaler_x.transform(x_test)