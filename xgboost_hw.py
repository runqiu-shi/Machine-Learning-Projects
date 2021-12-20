import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
plt.style.use('fivethirtyeight')

def create_features(df, label=None):
    """
    Creates time series features from datetime index
    """
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    #df['weekofyear'] = df['date'].dt.weekofyear
    df['weekofyear'] = df['date'].dt.isocalendar().week
    df['weekofyear'] = df['weekofyear'].astype(int)

    X = df[['hour', 'dayofweek', 'quarter', 'month', 'year',
            'dayofyear', 'dayofmonth', 'weekofyear']]
    if label:
        y = df[label]
        return X, y
    return X

def create_features_fcst(data):
    row = round(0.9 * data.shape[0])
    data = data.values
    train = data[:int(row), :]
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = data[int(row):, :-1]
    y_test = data[int(row):, -1]
   # x_train = np.reshape(x_train, (x_train.shape[0],1,1, x_train.shape[1]))
   # x_test = np.reshape(x_test, (x_test.shape[0],1,1, x_test.shape[1]))
    return x_train, y_train, x_test, y_test

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def power_sample():
    pjme = pd.read_csv('data/PJME_hourly.csv', index_col=[0], parse_dates=[0])
    color_pal = ["#F8766D", "#D39200", "#93AA00", "#00BA38", "#00C19F", "#00B9E3", "#619CFF", "#DB72FB"]
    _ = pjme.plot(style='.', figsize=(15, 5), color=color_pal[0], title='PJM East')

    split_date = '01-Jan-2015'
    pjme_train = pjme.loc[pjme.index <= split_date].copy()
    pjme_test = pjme.loc[pjme.index > split_date].copy()
    _ = pjme_test \
        .rename(columns={'PJME_MW': 'TEST SET'}) \
        .join(pjme_train.rename(columns={'PJME_MW': 'TRAINING SET'}), how='outer') \
        .plot(figsize=(15,5), title='PJM East', style='.')

    X_train, y_train = create_features(pjme_train, label='PJME_MW')
    X_test, y_test = create_features(pjme_test, label='PJME_MW')

    reg = xgb.XGBRegressor(n_estimators=1000)
    reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=50, verbose=True)#Change verbose to True if you want to see it train

    _ = plot_importance(reg, height=0.9)
    pjme_test['MW_Prediction'] = reg.predict(X_test)
    pjme_all = pd.concat([pjme_test, pjme_train], sort=False)

    _ = pjme_all[['PJME_MW','MW_Prediction']].plot(figsize=(15, 5))

    # Plot the forecast with the actuals
    f, ax = plt.subplots(1)
    f.set_figheight(5)
    f.set_figwidth(15)
    _ = pjme_all[['MW_Prediction','PJME_MW']].plot(ax=ax,
                                                  style=['-','.'])
    ax.set_xbound(lower='01-01-2015', upper='02-01-2015')
    ax.set_ylim(0, 60000)
    plot = plt.suptitle('January 2015 Forecast vs Actuals')

    # Plot the forecast with the actuals
    f, ax = plt.subplots(1)
    f.set_figheight(5)
    f.set_figwidth(15)
    _ = pjme_all[['MW_Prediction','PJME_MW']].plot(ax=ax,
                                                  style=['-','.'])
    ax.set_xbound(lower='01-01-2015', upper='01-08-2015')
    ax.set_ylim(0, 60000)
    plot = plt.suptitle('First Week of January Forecast vs Actuals')

    f, ax = plt.subplots(1)
    f.set_figheight(5)
    f.set_figwidth(15)
    _ = pjme_all[['MW_Prediction','PJME_MW']].plot(ax=ax,
                                                  style=['-','.'])
    ax.set_ylim(0, 60000)
    ax.set_xbound(lower='07-01-2015', upper='07-08-2015')
    plot = plt.suptitle('First Week of July Forecast vs Actuals')

    mean_squared_error(y_true=pjme_test['PJME_MW'],
                       y_pred=pjme_test['MW_Prediction'])

    mean_absolute_error(y_true=pjme_test['PJME_MW'],
                       y_pred=pjme_test['MW_Prediction'])

    mean_absolute_percentage_error(y_true=pjme_test['PJME_MW'],
                       y_pred=pjme_test['MW_Prediction'])

    pjme_test['error'] = pjme_test['PJME_MW'] - pjme_test['MW_Prediction']
    pjme_test['abs_error'] = pjme_test['error'].apply(np.abs)
    error_by_day = pjme_test.groupby(['year','month','dayofmonth']) \
        .mean()[['PJME_MW','MW_Prediction','error','abs_error']]
    error_by_day.sort_values('error', ascending=True).head(10)

    error_by_day.sort_values('abs_error', ascending=False).head(10)
    error_by_day.sort_values('abs_error', ascending=True).head(10)

    f, ax = plt.subplots(1)
    f.set_figheight(5)
    f.set_figwidth(10)
    _ = pjme_all[['MW_Prediction','PJME_MW']].plot(ax=ax,
                                                  style=['-','.'])
    ax.set_ylim(0, 60000)
    ax.set_xbound(lower='08-13-2016', upper='08-14-2016')
    plot = plt.suptitle('Aug 13, 2016 - Worst Predicted Day')

    f, ax = plt.subplots(1)
    f.set_figheight(5)
    f.set_figwidth(10)
    _ = pjme_all[['MW_Prediction','PJME_MW']].plot(ax=ax,
                                                  style=['-','.'])
    ax.set_ylim(0, 60000)
    ax.set_xbound(lower='10-03-2016', upper='10-04-2016')
    plot = plt.suptitle('Oct 3, 2016 - Best Predicted Day')

    f, ax = plt.subplots(1)
    f.set_figheight(5)
    f.set_figwidth(10)
    _ = pjme_all[['MW_Prediction','PJME_MW']].plot(ax=ax,
                                                  style=['-','.'])
    ax.set_ylim(0, 60000)
    ax.set_xbound(lower='08-13-2016', upper='08-14-2016')
    plot = plt.suptitle('Aug 13, 2016 - Worst Predicted Day')

if __name__ == '__main__':
    data = pd.read_csv('./data/demand_FCST.csv', )
    x_train, y_train, x_test, y_test = create_features_fcst(data)
    reg = xgb.XGBRegressor(n_estimators=1000)
    #x_train = pd.DataFrame(x_train, columns=['Year', 'Quarter', 'Month', 'A', 'B', 'C', 'D', 'E', 'F', 'Booking Quantity'])
    reg.fit(x_train,y_train,
            eval_set=[(x_train, y_train), (x_test, y_test)],
            early_stopping_rounds=50, verbose=True)#Change verbose to True if you want to see it train

    #reg.fit(pd.DataFrame(x_train,columns=['Year', 'Quarter', 'Month', 'A', 'B', 'C', 'D', 'E', 'F', 'Booking Quantity']),y_train)
    _ = plot_importance(reg, height=0.9)
    y_predict = reg.predict(x_test)

    print(mean_squared_error(y_test,y_predict))

    fig = plt.figure(facecolor='white')
    # plot result
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(y_test, label='True Data')
    ax.plot(y_predict, label='Predict')
    ax.legend()
    plt.show()
    plt.savefig("./cnn_lstm/xg_comp.jpg", dpi=fig.dpi)
    np.savetxt("./cnn_lstm/xgboost_truedata.txt", y_test)
    np.savetxt("./cnn_lstm/xgboost_predict.txt", y_predict)
    print(y_predict)