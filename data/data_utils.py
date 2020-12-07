import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pickle import dump
from pickle import load
import csv
import math


def load_training_data(sequence_length, percent_data_for_training, data_path, start_date):

    df = pd.read_csv(data_path, usecols=['Date', 'Open', 'High', 'Low', 'Close'],
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
    df = pd.read_csv(data_path, usecols=['Date', 'Open', 'High', 'Low', 'Close'], parse_dates=True)

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
        if index > 0:
            close_close = math.log(float(row[4]) / previous[4])
            open_open = math.log(float(row[1]) / previous[1])
            close_open = math.log(float(row[1]) / previous[4])
            open_close = math.log(float(row[4]) / float(row[1]))
            open_high = math.log(float(row[2]) / float(row[1]))
            close_high = math.log(float(row[2]) / float(row[4]))
            low_open = math.log(float(row[1]) / float(row[3]))
            low_close = math.log(float(row[4]) / float(row[3]))
            low_high = math.log(float(row[2]) / float(row[3]))
            volume = math.log(float(row[6]) / previous[5])
            processed.append([close_close, close_open])

        previous = [row[0], float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[6])]
        index += 1

    # remove index 0 to match processed above
    df = df[1:]

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

