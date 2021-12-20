# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 20:27:29 2018
@author: GY
"""
from sklearn import svm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from math import sqrt
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR



def data_pre(data, max_value, min_value, scaler):
    """
    :param data:
    :param seq_len:
    :return:
    """
    row = round(0.866 * data.shape[0])
    data3 = data.values
    x_train = data3[:int(row), :21]
    y_train = data3[:int(row), 21]
    x_test = data3[int(row):, :21]
    y_test = data3[int(row):, 21]
    print(y_test.shape)
    param_grid = {'kernel': ['poly'], 'C': np.logspace(-10, 5, 16), 'degree': np.arange(1, 4)}
    grid = GridSearchCV(SVR(), param_grid, scoring='neg_mean_squared_error')
    grid.fit(x_train, y_train)

    print("The best parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))
    #
    p3 = grid.predict(x_train)
    p4 = grid.predict(x_test)




    error2 = sqrt(mean_squared_error(y_train, p3))
    print('Polinomial model Test MSE: %.3f' % error2)
    error2 = sqrt(mean_squared_error(y_test, p4))
    print('Polinomial model Test MSE: %.3f' % error2)

    y_test_scaled = y_test * (max_value-min_value) + min_value
    p4_scaled = p4 * (max_value - min_value) + min_value

    np.savetxt('prediction with normalization.txt', p4_scaled)

    plt.plot(y_test_scaled, label='True Data', color='blue')
    plt.plot(p4_scaled, label='Prediction', color='red')
    plt.title('Support Vector Regression(Polynomial)')
    plt.xlabel("Monthly Data")
    plt.ylabel("Bill Quantity")
    plt.legend()

    plt.show()






if __name__ == "__main__":
    print('> Loading data... ')
    df = pd.read_csv('tmd4.csv', header=0)
    values = df.values.astype('float32')
    max_value = np.max(values[:,21])
    print(max_value)
    min_value = np.min(values[:,21])
    print(min_value)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(values)
    df = pd.DataFrame(scaled)
    print(df.values.shape)
    print(df.values)
    data_pre(df, max_value, min_value, scaler)

