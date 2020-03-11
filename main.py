# Winter Roberts
# CSE 163 AD with Andrew Frazier and Paul Pham
# Final Project
# Measures the ability of tweet sentiment and hourly
# stock price data to predict future prices.


import pandas as pd
import requests
from io import StringIO

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

url = "http://static.winterroberts.me/data/twitter_brands.csv"


def reduce_set(data):
    """
    Used by serveral analyses to reduce the dataset to stock
    symbol and model features.
    """
    return data.loc[:, ['stock_symbol', 'stock_price_last',
                        'tweet_volume', 'unix_diff',
                        'positivity_percentage', 'currency']]


def add_empty_currency_col(sdf):
    """
    Adds additional currency features to per-symbol dataframes
    that do not generate both columns by name.
    """
    if {'currency_USD'}.issubset(sdf.columns):
        sdf['currency_KRW'] = 0
    else:
        sdf['currency_USD'] = 0


def full_set_fit(data):
    """
    Trains and compares network models that differ by the number and nodes
    in and number of hidden layers, returning the most reliable and graphing
    their learning over 100 epochs.
    """
    print("Fitting...")
    x = pd.get_dummies(data.loc[:, ['stock_price_last', 'tweet_volume',
                                    'unix_diff', 'positivity_percentage',
                                    'currency']])
    y = data['stock_price']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    mlp_candidates = [(50,), (50, 25), (100,), (100, 50)]
    best_mlp = None
    best_score = 0
    per_partial = X_train.shape[0] // 100
    score_train = list()
    score_test = list()
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, figsize=(20, 10), ncols=2)
    # Try each mlp candidate and return the best (reliably (100,50)).
    for i in range(4):
        mlp = MLPRegressor(hidden_layer_sizes=mlp_candidates[i],
                           random_state=1)
        score_train.append(list())
        score_test.append(list())
        for epoch in range(100):
            mlp.partial_fit(X_train[epoch*per_partial+1:
                            epoch*per_partial+per_partial+1],
                            y_train[epoch*per_partial+1:
                            epoch*per_partial+per_partial+1])
            epoch_score_train = mlp.score(X_train, y_train)
            epoch_score_test = mlp.score(X_test, y_test)
            # Append partial training scores to show on graph
            score_train[i].append(epoch_score_train)
            score_test[i].append(epoch_score_test)

        mlp_score = mlp.score(X_test, y_test)
        print(mlp_score)
        if mlp_score > best_score:
            best_score = mlp_score
            best_mlp = mlp
            print("New best:", mlp_score, " Shape:", mlp_candidates[i])

    hundred_epoch = [i for i in range(100)]

    ax1.plot(hundred_epoch, score_train[0], label='training score')
    ax1.plot(hundred_epoch, score_test[0], label='testing score')
    ax1.set_xlabel("Training Epochs")
    ax1.set_ylabel("Accuracy Score")
    ax1.legend()
    ax1.set_title('Accuracy for shape (50,) per partial fit epoch')

    ax2.plot(hundred_epoch, score_train[1], label='training score')
    ax2.plot(hundred_epoch, score_test[1], label='testing score')
    ax2.set_xlabel("Training Epochs")
    ax2.set_ylabel("Accuracy Score")
    ax2.legend()
    ax2.set_title('Accuracy for shape (50,25) per partial fit epoch')

    ax3.plot(hundred_epoch, score_train[2], label='training score')
    ax3.plot(hundred_epoch, score_test[2], label='testing score')
    ax3.set_xlabel("Training Epochs")
    ax3.set_ylabel("Accuracy Score")
    ax3.legend()
    ax3.set_title('Accuracy for shape (100,) per partial fit epoch')

    ax4.plot(hundred_epoch, score_train[3], label='training score')
    ax4.plot(hundred_epoch, score_test[3], label='testing score')
    ax4.set_xlabel("Training Epochs")
    ax4.set_ylabel("Accuracy Score")
    ax4.legend()
    ax4.set_title('Accuracy for shape (100,50) per partial fit epoch')

    fig.savefig('accuracy_score_epochs.png')

    # Precautionary measure to ensure the network finds a local minimum.
    best_mlp.fit(X_train, y_train)
    return best_mlp


def predict_full_period(mlp, data):
    """
    Graphs the lowest and highest percent error for symbols in the dataset
    for the full period.
    """
    print("Full period low/high errors")
    datap = reduce_set(data)
    err_low = float(-1)
    err_high = float(0)
    for symbol in datap['stock_symbol'].unique():
        sdf = datap[datap['stock_symbol'] == symbol]
        sdf = pd.get_dummies(sdf.loc[:, sdf.columns != 'stock_symbol'])
        sdf = sdf.reset_index(drop=True)
        add_empty_currency_col(sdf)
        price_last = sdf.head(1)['stock_price_last']
        for i in range(len(sdf)):
            if len(sdf) <= i+1:
                continue
            row = sdf.loc[i+1, :]
            row['stock_price_last'] = price_last
            price_last = mlp.predict(row.to_numpy().reshape(1, -1))
        price_la = data[data['stock_symbol'] == symbol].tail(1)['stock_price']
        err_f = float(100*abs(price_last-price_la)/price_la)
        if err_low == -1:
            err_low = err_f
        if err_low > err_f:
            err_low = err_f
        if err_high < err_f:
            err_high = err_f
    plt.figure(figsize=(8, 4))
    plt.tick_params(axis='x', which='both', bottom=False, top=False,
                    labelbottom=False)
    plt.bar([1, 2], [err_low, err_high])
    plt.title('Error min/max for full-period prediction')
    plt.ylabel("Percent Error")
    plt.savefig('full_period_error.png')


def predict_interval(mlp, data):
    """
    Graphs the lowest and highest percent error for symbols in the dataset
    for each interval of any length.
    """
    print("Interval low/high errors")
    datap = reduce_set(data)
    err_low = float(-1)
    err_high = float(0)
    for symbol in datap['stock_symbol'].unique():
        sdf = datap[datap['stock_symbol'] == symbol]
        sdf = pd.get_dummies(sdf.loc[:, sdf.columns != 'stock_symbol'])
        sdf = sdf.reset_index(drop=True)
        add_empty_currency_col(sdf)
        for i in range(len(sdf)):
            if len(sdf) <= i+1:
                continue
            row = sdf.loc[i+1, :]
            price_pred = mlp.predict(row.to_numpy().reshape(1, -1))
            price_ac = data[data['stock_symbol'] == symbol]
            price_ac = price_ac.reset_index(drop=True)
            price_la = price_ac.loc[i+1, ['stock_price']]
            err_f = float(100*abs(price_pred-price_la)/price_la)
            if err_low == -1:
                err_low = err_f
            if err_low > err_f:
                err_low = err_f
            if err_high < err_f:
                err_high = err_f
    plt.figure(figsize=(8, 4))
    plt.tick_params(axis='x', which='both', bottom=False, top=False,
                    labelbottom=False)
    plt.bar([1, 2], [err_low, err_high])
    plt.title('Error min/max for interval prediction')
    plt.ylabel("Percent Error")
    plt.savefig('interval_error.png')


def predict_hourly(mlp, data):
    """
    Graphs the lowest and highest percent error for symbols in the dataset
    for each interval of an hour.
    """
    print("Hourly low/high errors")
    datap = reduce_set(data)
    err_low = float(-1)
    err_high = float(0)
    for symbol in datap['stock_symbol'].unique():
        sdf = datap[datap['stock_symbol'] == symbol]
        sdf = pd.get_dummies(sdf.loc[:, sdf.columns != 'stock_symbol'])
        sdf = sdf.reset_index(drop=True)
        add_empty_currency_col(sdf)
        for i in range(len(sdf)):
            if len(sdf) <= i+1 or int(sdf.loc[i+1, ['unix_diff']]) > 4000:
                continue
            row = sdf.loc[i+1, :]
            price_pred = mlp.predict(row.to_numpy().reshape(1, -1))
            price_ac = data[data['stock_symbol'] == symbol]
            price_ac = price_ac.reset_index(drop=True)
            price_la = price_ac.loc[i+1, ['stock_price']]
            err_f = float(100*abs(price_pred-price_la)/price_la)
            if err_low == -1:
                err_low = err_f
            if err_low > err_f:
                err_low = err_f
            if err_high < err_f:
                err_high = err_f
    plt.figure(figsize=(8, 4))
    plt.tick_params(axis='x', which='both', bottom=False, top=False,
                    labelbottom=False)
    plt.bar([1, 2], [err_low, err_high])
    plt.title('Error min/max for hourly prediction')
    plt.ylabel("Percent Error")
    plt.savefig('hourly_error.png')


def full_interval_direction(mlp, data):
    """
    Graphs the number of correct and incorrect predictions for stock
    price increases and decreases for all intervals.
    """
    print("Interval direction accuracy")
    datap = reduce_set(data)
    correct_up = 0
    correct_down = 0
    incorrect_up = 0
    incorrect_down = 0
    for symbol in datap['stock_symbol'].unique():
        sdf = datap[datap['stock_symbol'] == symbol]
        sdf = pd.get_dummies(sdf.loc[:, sdf.columns != 'stock_symbol'])
        sdf = sdf.reset_index(drop=True)
        add_empty_currency_col(sdf)
        for i in range(len(sdf)):
            if len(sdf) <= i+1:
                continue
            row = sdf.loc[i+1, :]
            price_pred = mlp.predict(row.to_numpy().reshape(1, -1))
            price_ac = data[data['stock_symbol'] == symbol]
            price_ac = price_ac.reset_index(drop=True)
            price_prev = float(price_ac.loc[i+1, ['stock_price_last']])
            price_la = float(price_ac.loc[i+1, ['stock_price']])
            if price_la > price_prev:
                # up
                if price_pred > price_prev:
                    correct_up += 1
                else:
                    incorrect_up += 1
            else:
                # down
                if price_pred <= price_prev:
                    correct_down += 1
                else:
                    incorrect_down += 1
    plt.figure(figsize=(8, 4))
    plt.tick_params(axis='x', which='both', bottom=False, top=False,
                    labelbottom=False)
    plt.bar([1, 2, 3, 4], [correct_up, correct_down, incorrect_up,
            incorrect_down])
    plt.title('Number of interval predicitions moving in the ' +
              'correct/incorrect direction.')
    plt.ylabel("Percent Error")
    plt.xlabel("Correct (up, down) / Incorrect (up, down)")
    plt.savefig('dir_interval.png')


def hourly_direction(mlp, data):
    """
    Graphs the number of correct and incorrect predictions for stock
    price increases and decreases for all hourly intervals.
    """
    print("Hourly direction accuracy")
    datap = reduce_set(data)
    correct_up = 0
    correct_down = 0
    incorrect_up = 0
    incorrect_down = 0
    for symbol in datap['stock_symbol'].unique():
        sdf = datap[datap['stock_symbol'] == symbol]
        sdf = pd.get_dummies(sdf.loc[:, sdf.columns != 'stock_symbol'])
        sdf = sdf.reset_index(drop=True)
        add_empty_currency_col(sdf)
        for i in range(len(sdf)):
            if len(sdf) <= i+1 or int(sdf.loc[i+1, ['unix_diff']]) > 4000:
                continue
            row = sdf.loc[i+1, :]
            price_pred = mlp.predict(row.to_numpy().reshape(1, -1))
            price_ac = data[data['stock_symbol'] == symbol]
            price_ac = price_ac.reset_index(drop=True)
            price_prev = float(price_ac.loc[i+1, ['stock_price_last']])
            price_la = float(price_ac.loc[i+1, ['stock_price']])
            if price_la > price_prev:
                # up
                if price_pred > price_prev:
                    correct_up += 1
                else:
                    incorrect_up += 1
            else:
                # down
                if price_pred <= price_prev:
                    correct_down += 1
                else:
                    incorrect_down += 1
    plt.figure(figsize=(8, 4))
    plt.tick_params(axis='x', which='both', bottom=False, top=False,
                    labelbottom=False)
    plt.bar([1, 2, 3, 4], [correct_up, correct_down, incorrect_up,
            incorrect_down])
    plt.title('Number of hourly predicitions moving in the ' +
              'correct/incorrect direction.')
    plt.ylabel("Percent Error")
    plt.xlabel("Correct (up, down) / Incorrect (up, down)")
    plt.savefig('dir_hourly.png')


def main():
    pd.set_option('mode.chained_assignment', None)
    s = requests.get(url).text
    print("Read Complete")
    df = pd.read_csv(StringIO(s))
    mlp = full_set_fit(df)
    predict_full_period(mlp, df)
    predict_interval(mlp, df)
    predict_hourly(mlp, df)
    full_interval_direction(mlp, df)
    hourly_direction(mlp, df)


if __name__ == "__main__":
    main()
