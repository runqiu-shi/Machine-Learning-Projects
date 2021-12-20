import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

data_true = pd.read_csv('./cnn_lstm/xgboost_truedata.txt', header=None)
data_xgb  = pd.read_csv('./cnn_lstm/xgboost_predict.txt', header=None)
data_cnn = pd.read_csv('./cnn/normalized_20_epoch/cnn_predict_norm_earlystopping.txt', header=None)
data_cnn_lstm_2000  = pd.read_csv('./cnn_lstm/2000_epoch/cnnlstm_predict.txt', header=None)
data_cnn_lstm_improved_1000  = pd.read_csv('./cnn_lstm/1000_epoch_improved/cnnlstm_predict.txt', header=None)
data_cnn_lstm_normal_200  = pd.read_csv('./cnn_lstm/normalized_200_epoch/cnnlstm_predict.txt', header=None)
data_cnn_lstm_improved_normal_450  = pd.read_csv('./cnn_lstm/normalized_450_epoch_improved/cnnlstm_predict.txt', header=None)
data_svr  = pd.read_csv('./cnn_lstm/prediction_without_normalization.txt', header=None)


fig = plt.figure(facecolor='white')
ax = fig.add_subplot(1, 1, 1)

ax.plot(data_true, label='True Data')
ax.plot(data_svr, label='SVR')
ax.plot(data_xgb, label='XGBoost')
ax.plot(data_cnn, label='CNN')
#ax.plot(data_cnn_lstm_2000, label='CNN LSTM 1')
#ax.plot(data_cnn_lstm_improved_1000, label='CNN LSTM 2')
ax.plot(data_cnn_lstm_normal_200, label='CNN LSTM 1')
ax.plot(data_cnn_lstm_improved_normal_450, label='CNN LSTM 2')

ax.legend(bbox_to_anchor=(0,1), loc='upper left')
plt.show()
plt.savefig("./cnn_lstm/plot_all.png", dpi=fig.dpi)

x = ['SVR','XGBoost','CNN','data_cnn_lstm_1','data_cnn_lstm_2']
y = []
y.append(mean_squared_error(data_svr,data_true))
y.append(mean_squared_error(data_xgb,data_true))
y.append(mean_squared_error(data_cnn,data_true))
#y.append(mean_squared_error(data_cnn_lstm_2000,data_true))
#y.append(mean_squared_error(data_cnn_lstm_improved_1000,data_true))
y.append(mean_squared_error(data_cnn_lstm_normal_200,data_true))
y.append(mean_squared_error(data_cnn_lstm_improved_normal_450,data_true))

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y)
plt.show()
plt.savefig("./cnn_lstm/mse_all.png", dpi=fig.dpi)

'''
print('xgboost:')
print(mean_squared_error(data_xgb,data_true))
print('data_cnn_lstm_2000:')
print(mean_squared_error(data_cnn_lstm_2000,data_true))
print('data_cnn_lstm_improved_1000:')
print(mean_squared_error(data_cnn_lstm_improved_1000,data_true))
print('data_cnn_lstm_normal_200:')
print(mean_squared_error(data_cnn_lstm_normal_200,data_true))
print('data_cnn_lstm_improved_normal_450:')
print(mean_squared_error(data_cnn_lstm_improved_normal_450,data_true))
'''

print('end')