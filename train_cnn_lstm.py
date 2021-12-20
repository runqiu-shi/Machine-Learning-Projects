# -*- coding: utf-8 -*-
"""
Created on Thu May 31 18:57:58 2018

@author: GY
"""
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yaml
import CNN_LSTM
import numpy as np


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    print('Start ploting')
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        if i < 10:
            continue
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()


# Main Run Thread
if __name__ == '__main__':
    epochs = 2000
    normalization = False#False#True

    load_model_only = False
    print('> Loading data... ')
    df = pd.read_csv('./data/demand_FCST_lstm.csv')#, header=None)
    #column = list(range(df.shape[1]))
    #column.remove(16)
    #column.append(16)
    #df = df.iloc[:, column]#ix

    #add normalization, 5.28
    y_train_max = None
    y_train_min = None
    y_test_orig = None
    if normalization:
        #add normalization test 5/28
        _, _, _, y_test_orig = CNN_LSTM.data_pre(df)
        values = df.values.astype('float32')
        y_train_max = np.max(values[:, -1],axis=0)
        y_train_min = np.min(values[:, -1],axis=0)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        df = pd.DataFrame(scaled)

    X_train, y_train, X_test, y_test = CNN_LSTM.data_pre(df)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # fit model
    global_start_time = time.time()
    print('> Data Loaded. Compiling...')

    history = None
    if not load_model_only:
        model = CNN_LSTM.build_model()
        #model = CNN_LSTM.build_model_improved()
        model, history = CNN_LSTM.fit_model(X_train, y_train, model, batch_size=16, nb_epoch=epochs, validation_split=0.2)#nb_epoch=1000,
        print('Training duration (s) : ', time.time() - global_start_time)

    # Predict
    predicted, score = CNN_LSTM.predict_point_by_point(X_test, y_test)
    if normalization:
        predicted = predicted * (y_train_max-y_train_min) + y_train_min
    print('Test score is: ', score)
    np.savetxt("./cnn_lstm/cnnlstm_predict.txt", predicted)

    #y = y_test*20446.87563+4975.270898
    #pre = predicted*20446.87563+4975.270898
    if normalization:
        y = y_test_orig
    else:
        y = y_test
    pre = predicted

    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(y, label='True Data')
    ax.plot(pre, label='Predict')
    ax.legend()
    plt.plot()
    plt.savefig("./cnn_lstm/cnn_lstm_comp.jpg", dpi=fig.dpi)
    plt.show()

   # add history plot 5/31
    if history is not None:
        print(history.history.keys())
        # summarize history for accuracy
        #plt.plot(history.history['accuracy'])
        #plt.plot(history.history['val_accuracy'])
        #plt.title('model accuracy')
        #plt.ylabel('accuracy')
        #plt.xlabel('epoch')
        #plt.legend(['train', 'test'], loc='upper left')
        #plt.show()
        # summarize history for loss
        plt.figure(2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'valid'], loc='upper right')
        plt.savefig("./cnn_lstm/cnn_history_loss_earlystopping.jpg", dpi=fig.dpi)
        plt.show()
    print('end')