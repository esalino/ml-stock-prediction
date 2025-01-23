import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from pickle import dump
from pickle import load
import csv
from datetime import datetime
import calendar
import pandas_ta as ta


def roll_up_daily_to_weekly(data_path):
    df = pd.read_csv(data_path, usecols=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], parse_dates=True)

    # Since we are using pandas we could skip csv iterator but iterating over a
    # pandas array has horrible performance
    infile = open(data_path)
    reader = csv.reader(infile)

    date_format = '%Y-%m-%d'

    start_date_found = False

    close_price_saved = []
    processed = []
    processed_y = []
    index = 0
    next(reader)
    for row in reader:
        date_obj = datetime.strptime(row[0], date_format)
        day_of_week_number = calendar.weekday(date_obj.year, date_obj.month, date_obj.day)

        if not start_date_found:
            if day_of_week_number != 0:
                continue
            else:
                start_date_found = True

        open_price = round(float(row[1]), 2)
        high_price = round(float(row[2]), 2)
        low_price = round(float(row[3]), 2)
        close_price = round(float(row[4]), 2)
        close_price_saved.append(close_price)
        volume = float(row[6])
        # we are starting at index once since using change from previous day
        if index >= index_start:
            close_close_weekly = (close_price / close_price_saved[index - 5]) - 1
            close_close = (close_price / close_price_saved[index - 1]) - 1
            open_close = (close_price / open_price) - 1

            open_close_mid = (open_price + close_price) / 2
            high_low_mid = (high_price + low_price) / 2

            mid_point_diff = (open_close_mid / high_low_mid) - 1

            moving_avg_volume_close = (volume / df.at[index, 'Volume Moving Avg']) - 1
            processed.append([close_close, open_close, moving_avg_volume_close, mid_point_diff])
            processed_y.append([close_close_weekly])

        index += 1

    # remove index 0 to match processed above
    df = df[index_start:]

    processed = np.array(processed)
    processed_y = np.array(processed_y)

    df['Close Log Dif'] = processed[:, 0]

    max_abs_scaler = MaxAbsScaler(copy=False)
    processed = max_abs_scaler.fit_transform(processed)
    processed_y = max_abs_scaler.fit_transform(processed_y)

    # create sliding windows of sequence length
    # expects target (close) at index 0
    def sliding_windows(data, data_y, seq_length):
        x = []
        y = []

        for i in range(len(data) - seq_length):
            _x = data[i:(i + seq_length)]
            # _y = data_y[i:(i + seq_length)]
            _y = [data_y[i + seq_length][0]]
            x.append(_x)
            y.append(_y)

        return np.array(x), np.array(y)

    train_data_x, train_data_y = sliding_windows(processed, processed_y, sequence_length)
    train_size = int(len(train_data_x) * percent_data_for_training)

    x_train_np = train_data_x[0:train_size]
    x_eval_np = train_data_x[train_size:len(train_data_x)]
    y_train_np = train_data_y[0:train_size]

    if percent_data_for_training == 1:
        x_eval_np = None
        dump(max_abs_scaler, open('../../saved_models/data_scaler.pkl', 'wb'))

    return x_train_np, x_eval_np, y_train_np, df


def load_training_data_weekly(sequence_length, percent_data_for_training, data_path):
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
    processed_y = []
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
            processed_y.append([close_close_5])

        previous.append([row[0], open_price, high_price, low_price, close_price, volume])
        index += 1

    # remove index 0 to match processed above
    df = df[index_start:]

    processed = np.array(processed)
    processed_y = np.array(processed_y)

    # df['Close Log Dif'] = processed[:, 0]

    max_abs_scaler = MaxAbsScaler(copy=False)
    processed = max_abs_scaler.fit_transform(processed)
    processed_y = max_abs_scaler.fit_transform(processed_y)

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

    train_data_x, train_data_y = sliding_windows(processed, processed_y, sequence_length)
    train_size = int(len(train_data_x) * percent_data_for_training)

    x_train_np = train_data_x[0:train_size]
    x_eval_np = train_data_x[train_size:len(train_data_x)]
    y_train_np = train_data_y[0:train_size]

    if percent_data_for_training == 1:
        x_eval_np = None
        dump(max_abs_scaler, open('.saved_models/data_scaler_weekly.pkl', 'wb'))

    return x_train_np, x_eval_np, y_train_np, df


def load_training_data_monthly(sequence_length, percent_data_for_training):
    data_path = "./raw_data/SPY.csv"
    data_path_tnx = "./raw_data/^TNX.csv"
    index_start = 50
    df = pd.read_csv(data_path, usecols=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], parse_dates=True)
    df_tnx = pd.read_csv(data_path_tnx, usecols=['Date', 'Open', 'High', 'Low', 'Close'], index_col='Date', parse_dates=True)
    df_tnx = df_tnx.dropna()

    df['Volume Moving Avg'] = df['Volume'].rolling(window=index_start).mean()
    df_tnx['Close Moving 10 Year'] = df_tnx['Close'].rolling(window=index_start).mean()

    df_tnx['Close Moving Avg Diff'] = df_tnx.apply(lambda row: (row["Close"] / row["Close Moving 10 Year"]) - 1, axis=1)
    df_tnx = df_tnx.loc['1993-04-12':]
    # Since we are using pandas we could skip csv iterator but iterating over a
    # pandas array has horrible performance
    infile = open(data_path)
    reader = csv.reader(infile)

    previous = []
    processed = []
    index = 0
    next(reader)
    for row in reader:
        open_price = round(float(row[1]), 2)
        high_price = round(float(row[2]), 2)
        low_price = round(float(row[3]), 2)
        close_price = round(float(row[4]), 2)
        volume = float(row[6])
        # we are starting at index once since using change from previous day
        if index >= index_start:
            close_close = (close_price / previous[4]) - 1
            open_close = (close_price / open_price) - 1
            open_close_mid = (open_price + close_price) / 2
            high_low_mid = (high_price + low_price) / 2

            mid_point_diff = (open_close_mid / high_low_mid) - 1

            moving_avg_volume_close = (volume / df.at[index, 'Volume Moving Avg']) - 1
            i_temp = index
            while df.at[i_temp, 'Date'] not in df_tnx.index:
                i_temp -= 1
            ten_year = df_tnx.at[df.at[i_temp, 'Date'], 'Close']

            processed.append([close_close, open_close, moving_avg_volume_close, mid_point_diff])

        previous = [row[0], open_price, high_price, low_price, close_price, volume]
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
            if i + seq_length + 20 >= len(data):
                break
            _x = data[i:(i + seq_length)]
            # we are predicting 20 days ahead
            _y = [data[i + seq_length + 20][0]]
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
        dump(max_abs_scaler, open('../../saved_models/data_scaler.pkl', 'wb'))

    return x_train_np, x_eval_np, y_train_np, df


def load_prediction_data_v2(sequence_length, data_path):

    df = pd.read_csv(data_path, usecols=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], parse_dates=True)
    index_start = df.shape[0] - 50

    df['Volume Moving Avg'] = df['Volume'].rolling(window=index_start).mean()

    # Since we are using pandas we could skip csv iterator but iterating over a
    # pandas array has horrible performance
    infile = open(data_path)
    reader = csv.reader(infile)

    previous = []
    processed = []
    index = 0
    next(reader)
    for row in reader:
        open_price = round(float(row[1]), 2)
        high_price = round(float(row[2]), 2)
        low_price = round(float(row[3]), 2)
        close_price = round(float(row[4]), 2)
        volume = float(row[6])
        # we are starting at index once since using change from previous day
        if index >= index_start:
            close_close = (close_price / previous[4]) - 1
            open_close = (close_price / open_price) - 1

            open_close_mid = (open_price + close_price) / 2
            high_low_mid = (high_price + low_price) / 2

            mid_point_diff = (open_close_mid / high_low_mid) - 1

            moving_avg_volume_close = (volume / df.at[index, 'Volume Moving Avg']) - 1
            processed.append([close_close, open_close, moving_avg_volume_close, mid_point_diff])

        previous = [row[0], open_price, high_price, low_price, close_price, volume]
        index += 1

    predict_data = np.array(processed)

    data_scaler = load(open('../../saved_models/data_scaler.pkl', 'rb'))
    predict_data_normalized = data_scaler.transform(predict_data)

    predict_sequence = [predict_data_normalized[len(predict_data_normalized) - sequence_length:len(predict_data_normalized)]]

    predict_sequence = np.array(predict_sequence)

    return predict_sequence, predict_data_normalized, data_scaler
