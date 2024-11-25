import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from pickle import dump
from pickle import load
import csv
import pandas_ta as ta


def get_processed_data(data_path):
    index_start = 50
    df = pd.read_csv(data_path, usecols=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], parse_dates=True)

    df.ta.rsi(length=14, append=True)
    df.ta.macd(append=True)
    df['Volume Moving Avg'] = df['Volume'].rolling(window=index_start).mean()

    # Since we are using pandas we could skip csv iterator but iterating over a
    # pandas array has horrible performance
    infile = open(data_path)
    reader = csv.reader(infile)

    previous = []
    processed = []
    processed_close_5 = []
    index = 0
    next(reader)
    for row in reader:
        open_price = round(float(row[1]), 2)
        high_price = round(float(row[2]), 2)
        low_price = round(float(row[3]), 2)
        close_price = round(float(row[4]), 2)
        volume = float(row[6])
        rsi_14 = df.at[index, 'RSI_14']
        # roc_10 = df.at[index, 'ROC_10']
        # we are starting at index once since using change from previous day
        if index >= index_start:
            close_close = (close_price / previous[index - 1][4]) - 1
            close_close_5 = (close_price / previous[index - 5][4]) - 1
            open_close = (close_price / open_price) - 1
            open_close_mid = (open_price + close_price) / 2
            high_low_mid = (high_price + low_price) / 2
            mid_point_diff = (open_close_mid / high_low_mid) - 1

            moving_avg_volume_close = (volume / df.at[index, 'Volume Moving Avg']) - 1

            processed.append([close_close, open_close, moving_avg_volume_close, mid_point_diff, rsi_14])
            processed_close_5.append([close_close_5])

        previous.append([row[0], open_price, high_price, low_price, close_price, volume])
        index += 1

    return (df, processed, processed_close_5)


def load_training_data_daily(sequence_length, percent_data_for_training, data_path):
    index_start = 50
    (df, processed, processed_close_5) = get_processed_data(data_path)

    # remove index 0 to match processed above
    df = df[index_start:]

    processed = np.array(processed)

    # df['Close Log Dif'] = processed[:, 0]

    max_abs_scaler_close = MaxAbsScaler(copy=False)
    first_elements = processed[:, 0]
    first_elements = first_elements.reshape(-1, 1)
    max_abs_scaler_close.fit_transform(first_elements)
    max_abs_scaler_processed = MaxAbsScaler(copy=False)
    processed = max_abs_scaler_processed.fit_transform(processed)

    # create sliding windows of sequence length
    # expects target (close) at index 0
    def sliding_windows(data, seq_length):
        x = []
        y = []

        for i in range(len(data) - seq_length):
            _x = data[i:(i + seq_length)]
            _y = [data[i + seq_length][0]]
            x.append(_x)
            y.append(_y)

        return np.array(x), np.array(y)

    train_data_x, train_data_y = sliding_windows(processed, sequence_length)
    train_size = int(len(train_data_x) * percent_data_for_training)

    x_train_np = train_data_x[0:train_size]
    x_eval_np = train_data_x[train_size:len(train_data_x) + 1]
    y_train_np = train_data_y[0:train_size]

    if percent_data_for_training == 1:
        x_eval_np = None
        dump(max_abs_scaler_processed, open('saved_models/data_scaler_daily.pkl', 'wb'))
        dump(max_abs_scaler_close, open('saved_models/close_scaler_daily.pkl', 'wb'))

    return x_train_np, x_eval_np, y_train_np, df


def load_prediction_data_daily(sequence_length, data_path):
    (df, processed) = get_processed_data(data_path)

    predict_data = np.array(processed)

    data_scaler = load(open('./saved_models/data_scaler_daily.pkl', 'rb'))
    predict_data_normalized = data_scaler.transform(predict_data)

    predict_sequence = [predict_data_normalized[len(predict_data_normalized) - sequence_length:len(predict_data_normalized)]]

    predict_sequence = np.array(predict_sequence)

    return predict_sequence, predict_data_normalized, data_scaler


def load_training_data_weekly(sequence_length, percent_data_for_training, data_path):
    index_start = 50

    (df, processed, processed_close_5) = get_processed_data(data_path)

    # remove index 0 to match processed above
    df = df[index_start:]

    processed = np.array(processed)
    processed_close_5 = np.array(processed_close_5)

    # df['Close Log Dif'] = processed[:, 0]

    max_abs_scaler_close = MaxAbsScaler(copy=False)
    first_elements = processed[:, 0]
    first_elements = first_elements.reshape(-1, 1)
    max_abs_scaler_close.fit_transform(first_elements)
    max_abs_scaler_processed = MaxAbsScaler(copy=False)
    processed = max_abs_scaler_processed.fit_transform(processed)

    # create sliding windows of sequence length
    # expects target (close) at index 0
    def sliding_windows(data, data_y, seq_length):
        x = []
        y = []

        for i in range(len(data) - seq_length - 4):
            _x = data[i:(i + seq_length)]
            _y = [data_y[i + 4 + seq_length][0]]
            x.append(_x)
            y.append(_y)

        return np.array(x), np.array(y)

    train_data_x, train_data_y = sliding_windows(processed, processed_close_5, sequence_length)
    train_size = int(len(train_data_x) * percent_data_for_training)

    x_train_np = train_data_x[0:train_size]
    x_eval_np = train_data_x[train_size:len(train_data_x)]
    y_train_np = train_data_y[0:train_size]

    if percent_data_for_training == 1:
        x_eval_np = None
        dump(max_abs_scaler_processed, open('saved_models/data_scaler_weekly.pkl', 'wb'))
        dump(max_abs_scaler_close, open('saved_models/close_scaler_weekly.pkl', 'wb'))

    return x_train_np, x_eval_np, y_train_np, df


def load_prediction_data_weekly(sequence_length, data_path):
    (df, processed, processed_close_5) = get_processed_data(data_path)

    predict_data = np.array(processed)

    data_scaler = load(open('./saved_models/data_scaler_weekly.pkl', 'rb'))
    predict_data_normalized = data_scaler.transform(predict_data)

    predict_sequence = [predict_data_normalized[len(predict_data_normalized) - sequence_length:len(predict_data_normalized)]]

    predict_sequence = np.array(predict_sequence)

    return predict_sequence, predict_data_normalized, data_scaler
