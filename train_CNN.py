# -*- coding: utf-8 -*-
"""
Created on Thu May 31 15:46:25 2018

@author: GY
"""
import time
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import CNN
from sklearn.preprocessing import MinMaxScaler
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
    epochs = 1000
    normalization_old = True#True

    normalization = False#obsolete

    print('> Loading data... ')
    df = pd.read_csv('./data/demand_FCST_lstm.csv')#./data/data(1).csv, header=None
    #column = list(range(df.shape[1]))
    #column.remove(16)
    #column.append(16)
    #df = df.iloc[:, column]
    y_train_max = None
    y_train_min = None
    y_test_orig = None
    if normalization_old:
        #add normalization test 5/28
        _, _, _, y_test_orig = CNN.data_pre(df, normalization)
        values = df.values.astype('float32')
        y_train_max = np.max(values[:, -1],axis=0)
        y_train_min = np.min(values[:, -1],axis=0)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        df = pd.DataFrame(scaled)


    X_train, y_train, X_test, y_test = CNN.data_pre(df,normalization)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    global_start_time = time.time()
    print('> Data Loaded. Compiling...')

    history = None
    # build and fit model
    model = CNN.build_model()
    model, history = CNN.fit_model(X_train, y_train, model, batch_size=16, nb_epoch=epochs, validation_split=0.2)#nb_epoch=1000,
    print('Training duration (s) : ', time.time() - global_start_time)

    # Predict
    predicted, score = CNN.predict_point_by_point(X_test, y_test)
    print('Test score is: ', score)

    if normalization or normalization_old:
        predicted = predicted * (y_train_max-y_train_min) + y_train_min
    np.savetxt("./cnn/cnn_predict_norm_earlystopping.txt", predicted)
    
    fig = plt.figure(facecolor='white')

    # plot result
    ax = fig.add_subplot(1, 1, 1)

    if normalization_old:
        ax.plot(y_test_orig, label='True Data')#y_test
    else:
        ax.plot(y_test, label='True Data')  # y_test

    ax.plot(predicted, label='Predict')
    ax.legend()
    plt.savefig("./cnn/cnn_predict_norm_earlystopping.jpg", dpi=fig.dpi)
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
        plt.savefig("./cnn/cnn_history_loss_earlystopping.jpg", dpi=fig.dpi)
        plt.show()
    print('end')
