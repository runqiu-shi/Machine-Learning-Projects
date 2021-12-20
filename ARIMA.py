# grid search ARIMA parameters for time series
import warnings
from math import sqrt
from pandas import read_csv
import pandas as pd
from pandas import datetime
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pylab import *


# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.7)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    rmse = sqrt(mean_squared_error(test, predictions))
    return rmse


# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
	row_num = len(p_values) * len(d_values) * len(q_values)
	column_num = 4
	parameter_result_store = np.zeros((row_num, column_num))
	i = 0
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				parameter_result_store[i, 0] = p
				parameter_result_store[i, 1] = d
				parameter_result_store[i, 2] = q

				order = (p,d,q)

				try:
					rmse = evaluate_arima_model(dataset, order)
					parameter_result_store[i, 3] = rmse
					# print(parameter_result_store)
					i += 1
					if rmse < best_score:
						best_score, best_cfg = rmse, order
					print('ARIMA%s RMSE=%.3f' % (order,rmse))
				except:
					continue
	print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))

	print(np.shape(best_cfg))
	print(type(best_cfg))

	return [parameter_result_store, best_cfg]

def predicion_models(X, best_cfg, max_value, min_value):
	train_size = int(len(X) * 0.7)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=best_cfg)
		model_fit = model.fit()
		output = model_fit.forecast()
		print(output)
		yhat = output[0]
		predictions.append(yhat)
		obs = test[t]
		history.append(obs)
		print(f'predicted={yhat}, expected={obs}')

	test_scaled = test * (max_value - min_value) + min_value
	predictions_scaled = predictions * (max_value - min_value) + min_value

	rmse = sqrt(mean_squared_error(test, predictions))
	print(f'Test RMSE: {rmse}')
	plt.plot(test_scaled, label='True data', color='blue')
	plt.plot(predictions_scaled, label='Prediction', color='red')
	plt.title('ARIMA')
	plt.xlabel("Monthly Data")
	plt.ylabel("Bill Quantity")

	plt.legend()
	plt.show()



def main():
	df = pd.read_csv('demand FCST 3.3.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
	values = df.values.astype('float32').reshape(-1,1)
	print(values.shape)

	max_value = max(values)
	print(max_value)
	min_value = min(values)
	print(min_value)

	scaler = MinMaxScaler(feature_range=(0, 1))
	scaled = scaler.fit_transform(values)
	df = pd.DataFrame(scaled)
	print(df.values.shape)
	print(df.values)


	# evaluate parameters  如果要test的话，需要把数值改小
	p_values = np.arange(1, 4)
	d_values = range(0, 2)
	q_values = range(0, 4)
	warnings.filterwarnings("ignore")
	parameter_result_store, best_cfg = evaluate_models(df.values, p_values, d_values, q_values)

	predicion_models(df.values, best_cfg, max_value, min_value)


	fig = plt.figure(figsize=(8, 6))
	ax = fig.add_subplot(111, projection='3d')



	colmap = cm.ScalarMappable(cmap=cm.hot)
	colmap.set_array(parameter_result_store[:, 3])


	ax.scatter(parameter_result_store[:, 0], parameter_result_store[:, 1], parameter_result_store[:, 2], marker='s', s=140,
			   c=parameter_result_store[:, 3], cmap='hot');
	cb = fig.colorbar(colmap)

	ax.set_xlabel('p');
	ax.set_ylabel('d');
	ax.set_zlabel('q');
	plt.show()


if __name__ == "__main__":
    main()