import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from pickle import dump
from pickle import load
import csv
import math


def load_training_data(sequence_length, percent_data_for_training, data_path, start_date):

    # df = pd.read_csv(data_path, usecols=['Date', 'Open', 'High', 'Low', 'Close'],
    #                 index_col='Date', parse_dates=True)

    df = pd.read_csv(data_path, usecols=['Date', 'Open', 'Close'],
                                      index_col='Date', parse_dates=True)

    df = df.loc[start_date:]

    df.reset_index(drop=False, inplace=True)
    #df['Day Of Week'] = df.apply(lambda row: row['Date'].weekday(), axis=1)
    df = df.drop('Date', 1)

    train_close = df['Close'].to_numpy()
    scalar_close = MinMaxScaler(feature_range=(0, 1))
    scalar_close.fit_transform(train_close.reshape(-1, 1))

    train_data = df.to_numpy()

    data_scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_full_data_set = data_scaler.fit_transform(train_data)

    def sliding_windows(data, seq_length):
        x = []
        y = []

        for i in range(len(data) - seq_length):
            _x = data[i:(i + seq_length)]
            _y = [data[i + seq_length][0]]
            x.append(_x)
            y.append(_y)

        return np.array(x), np.array(y)

    train_data_x, train_data_y = sliding_windows(normalized_full_data_set, sequence_length)
    train_size = int(len(train_data_x) * percent_data_for_training)

    x_train_np = train_data_x[0:train_size]
    x_eval_np = train_data_x[train_size:len(train_data_x)]
    y_train_np = train_data_y[0:train_size]

    if percent_data_for_training == 1:
        x_eval_np = None
        dump(data_scaler, open('../../saved_models/data_scaler.pkl', 'wb'))
        dump(scalar_close, open('../../saved_models/close_scaler.pkl', 'wb'))

    return x_train_np, x_eval_np, y_train_np, normalized_full_data_set, data_scaler, scalar_close


def load_training_data_v2(sequence_length, percent_data_for_training, data_path):
    index_start = 50
    df = pd.read_csv(data_path, usecols=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], parse_dates=True)

    df['Volume Moving Avg'] = df['Volume'].rolling(window=50).mean()

    # Since we are using pandas we could skip csv iterator but iterating over a
    # pandas array has horrible performance
    infile = open(data_path)
    reader = csv.reader(infile)

    previous = []
    processed = []
    index = 0
    next(reader)
    for row in reader:
        # we are starting at index once since using change from previous day
        if index >= index_start:
            close_close = ((float(row[4]) / previous[4]) - 1)
            open_open = ((float(row[1]) / previous[1]) - 1)
            high_high = ((float(row[2]) / previous[2]) - 1)
            low_low = ((float(row[3]) / previous[3]) - 1)
            close_open = ((float(row[1]) / previous[4]) - 1)
            open_close = ((float(row[4]) / float(row[1])) - 1)
            open_high = ((float(row[2]) / float(row[1])) - 1)
            close_high = ((float(row[2]) / float(row[4])) - 1)
            low_open = ((float(row[1]) / float(row[3])) - 1)
            close_low = ((float(row[3]) / float(row[4])) - 1)
            low_high = ((float(row[2]) / float(row[3])) - 1)
            volume = ((float(row[6]) / previous[5]) - 1)

            open_close_mid = (float(row[1]) + float(row[4])) / 2
            high_low_mid = (float(row[2]) + float(row[3])) / 2

            mid_point_diff = (open_close_mid/high_low_mid) - 1

            #five_day_moving_close = (float(row[4]) / df.loc[index, ['Five Day Moving']]) - 1
            moving_avg_volume_close = (float(row[5]) / df.loc[index, ['Volume Moving Avg']]) - 1
            processed.append([close_close, open_close, moving_avg_volume_close, mid_point_diff])

        previous = [row[0], float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[6])]
        index += 1

    # remove index 0 to match processed above
    df = df[index_start:]

    processed = np.array(processed)

    df['Close Log Dif'] = processed[:, 0]

    max_abs_scaler = MaxAbsScaler(copy=False)
    processed = max_abs_scaler.fit_transform(processed)

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
    x_eval_np = train_data_x[train_size:len(train_data_x)]
    y_train_np = train_data_y[0:train_size]

    if percent_data_for_training == 1:
        x_eval_np = None

    return x_train_np, x_eval_np, y_train_np, df


def load_prediction_data(sequence_length, data_path):

    df = pd.read_csv(data_path, usecols=['Close', 'Open', 'High', 'Low'])

    predict_data = df.to_numpy()

    data_scaler = load(open('../../saved_models/data_scaler.pkl', 'rb'))
    predict_data_normalized = data_scaler.fit_transform(predict_data)

    predict_sequence = [predict_data_normalized[len(predict_data_normalized) - sequence_length:len(predict_data_normalized)]]

    predict_sequence = np.array(predict_sequence)

    return predict_sequence, predict_data_normalized, data_scaler

