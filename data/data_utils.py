import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from pickle import dump
from pickle import load
import csv
import pandas_ta as ta


def sliding_windows_look_ahead(data, seq_length, data_y_for_look_ahead=None, days_look_ahead=1):
    assert (days_look_ahead == 1 and data_y_for_look_ahead is None) or (days_look_ahead > 1 and data_y_for_look_ahead is not None)
    x = []
    y = []
    data_y = data_y_for_look_ahead if days_look_ahead > 1 else data

    for i in range(len(data) - seq_length - (days_look_ahead - 1)):
        _x = data[i:(i + seq_length)]
        _y = [data_y[i + seq_length + (days_look_ahead - 1)][0]]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)


def get_processed_data(data_path):
    index_start = 50
    df = pd.read_csv(data_path, usecols=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], parse_dates=True)
    df = df.round(2)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(append=True)
    df.ta.sma(length=5, append=True)
    df.ta.sma(length=10, append=True)
    df.ta.sma(length=20, append=True)
    df.ta.sma(length=50, append=True)
    # df.ta.sma(length=100, append=True)
    df.ta.ha(append=True)
    df['Volume Moving Avg'] = df['Volume'].rolling(window=index_start).mean()

    # Since we are using pandas we could skip csv iterator but iterating over a
    # pandas array has horrible performance
    infile = open(data_path)
    reader = csv.reader(infile)

    previous = []
    processed = []
    processed_close_5 = []
    processed_close_10 = []
    processed_close_20 = []
    index = 0
    next(reader)
    for row in reader:
        open_price = round(float(row[1]), 2)
        high_price = round(float(row[2]), 2)
        low_price = round(float(row[3]), 2)
        close_price = round(float(row[4]), 2)
        volume = float(row[6])
        rsi_14 = df.at[index, 'RSI_14']
        sma_20 = df.at[index, 'SMA_20']
        sma_50 = df.at[index, 'SMA_50']
        MACDh_12_26_9 = df.at[index, 'MACDh_12_26_9']
        HA_open = df.at[index, 'HA_open']
        HA_high = df.at[index, 'HA_high']
        HA_low = df.at[index, 'HA_low']
        HA_close = df.at[index, 'HA_close']
        ha_close_over_open = (HA_close / HA_open) - 1
        ha_high_diff = (HA_high / HA_close) - 1 if ha_close_over_open >= 0 else (HA_high / HA_open) - 1
        ha_low_diff = (HA_low / HA_open) - 1 if ha_close_over_open >= 0 else (HA_low / HA_close) - 1

        if index >= index_start:
            close_close = (close_price / previous[index - 1][4]) - 1
            close_close_5 = (close_price / previous[index - 4][4]) - 1
            close_close_10 = (close_price / previous[index - 9][4]) - 1
            close_close_20 = (close_price / previous[index - 19][4]) - 1
            open_close = (close_price / open_price) - 1
            open_close_mid = (open_price + close_price) / 2
            high_low_mid = (high_price + low_price) / 2
            mid_point_diff = (open_close_mid / high_low_mid) - 1

            volume_over_volume_sma_50 = (volume / df.at[index, 'Volume Moving Avg']) - 1
            rsi_change = (rsi_14 / previous[index - 1][6]) - 1
            sma_20_over_50 = (sma_20 / sma_50) - 1

            processed.append([close_close, open_close, volume_over_volume_sma_50,
                              mid_point_diff, rsi_change, rsi_14, sma_20_over_50,
                              MACDh_12_26_9, HA_open, HA_high, HA_low, HA_close, ha_close_over_open, ha_high_diff, ha_low_diff])

            processed_close_5.append([close_close_5])
            processed_close_10.append([close_close_10])
            processed_close_20.append([close_close_20])

        previous.append([row[0], open_price, high_price, low_price, close_price, volume, rsi_14])
        index += 1

    # remove index 0 to match processed above
    df = df[index_start:]
    return (df, np.array(processed), np.array(processed_close_5), np.array(processed_close_10), np.array(processed_close_20))


def load_training_data_daily(sequence_length, percent_data_for_training, data_path):
    (df, processed, processed_close_5, processed_close_10, processed_close_20) = get_processed_data(data_path)

    # Delete unwanted data points from processed
    processed = np.delete(processed, (7), axis=1)  # MACDh_12_26_9
    processed = np.delete(processed, (6), axis=1)  # sma_20_over_50
    processed = np.delete(processed, (5), axis=1)  # rsi_14
    # processed = np.delete(processed, (4), axis=1)  # rsi_change
    processed = np.delete(processed, (3), axis=1)  # mid_point_diff
    # processed = np.delete(processed, (2), axis=1)  # volume_over_volume_sma_50
    # processed = np.delete(processed, (1), axis=1)  # open_close
    # processed = np.delete(processed, (0), axis=1)  # close_close

    # df['Close Log Dif'] = processed[:, 0]

    max_abs_scaler_close = MaxAbsScaler(copy=False)
    first_elements = processed[:, 0]
    first_elements = first_elements.reshape(-1, 1)
    max_abs_scaler_close.fit_transform(first_elements)
    max_abs_scaler_processed = MaxAbsScaler(copy=False)
    processed = max_abs_scaler_processed.fit_transform(processed)

    train_data_x, train_data_y = sliding_windows_look_ahead(processed, sequence_length)
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
    (df, processed, processed_close_5, processed_close_10, processed_close_20) = get_processed_data(data_path)

    data_scaler = load(open('./saved_models/data_scaler_daily.pkl', 'rb'))
    predict_data_normalized = data_scaler.transform(processed)

    predict_sequence = [predict_data_normalized[len(predict_data_normalized) - sequence_length:len(predict_data_normalized)]]

    predict_sequence = np.array(predict_sequence)

    return predict_sequence, predict_data_normalized, data_scaler


def load_training_data_weekly(sequence_length, percent_data_for_training, data_path):
    (df, processed, processed_close_5, processed_close_10, processed_close_20) = get_processed_data(data_path)

    # df['Close Log Dif'] = processed[:, 0]

    # Delete unwanted data points from processed
    # processed = np.delete(processed, (11), axis=1)  # HA_close
    # processed = np.delete(processed, (10), axis=1)  # HA_low
    # processed = np.delete(processed, (9), axis=1)  # HA_high
    # processed = np.delete(processed, (8), axis=1)  # HA_open
    processed = np.delete(processed, (7), axis=1)  # MACDh_12_26_9
    processed = np.delete(processed, (6), axis=1)  # sma_20_over_50
    # processed = np.delete(processed, (5), axis=1)  # rsi_14

    processed = np.delete(processed, (4), axis=1)  # rsi_change
    processed = np.delete(processed, (3), axis=1)  # mid_point_diff

    processed = np.delete(processed, (2), axis=1)  # volume_over_volume_sma_50

    processed = np.delete(processed, (1), axis=1)  # open_close

    processed = np.delete(processed, (0), axis=1)  # close_close

    max_abs_scaler_close = MaxAbsScaler(copy=False)
    first_elements = processed[:, 0]
    first_elements = first_elements.reshape(-1, 1)
    max_abs_scaler_close.fit_transform(first_elements)
    max_abs_scaler_processed = MaxAbsScaler(copy=False)
    processed = max_abs_scaler_processed.fit_transform(processed)
    max_abs_scaler_processed_closed = MaxAbsScaler(copy=False)
    processed_close_5 = max_abs_scaler_processed_closed.fit_transform(processed_close_5)

    train_data_x, train_data_y = sliding_windows_look_ahead(processed, sequence_length, processed_close_5, 5)
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
    (df, processed, processed_close_5, processed_close_10, processed_close_20) = get_processed_data(data_path)

    data_scaler = load(open('./saved_models/data_scaler_weekly.pkl', 'rb'))
    predict_data_normalized = data_scaler.transform(processed)

    predict_sequence = [predict_data_normalized[len(predict_data_normalized) - sequence_length:len(predict_data_normalized)]]

    predict_sequence = np.array(predict_sequence)

    return predict_sequence, predict_data_normalized, data_scaler


def load_training_data_biweekly(sequence_length, percent_data_for_training, data_path):
    (df, processed, processed_close_5, processed_close_10, processed_close_20) = get_processed_data(data_path)

    # df['Close Log Dif'] = processed[:, 0]

    # Delete unwanted data points from processed
    # processed = np.delete(processed, (14), axis=1)  # ha_low_diff
    # processed = np.delete(processed, (13), axis=1)  # ha_high_diff
    # processed = np.delete(processed, (12), axis=1)  # ha_close_over_open
    processed = np.delete(processed, (11), axis=1)  # HA_close
    processed = np.delete(processed, (10), axis=1)  # HA_low
    processed = np.delete(processed, (9), axis=1)  # HA_high
    processed = np.delete(processed, (8), axis=1)  # HA_open
    processed = np.delete(processed, (7), axis=1)  # MACDh_12_26_9
    processed = np.delete(processed, (6), axis=1)  # sma_20_over_50
    # processed = np.delete(processed, (5), axis=1)  # rsi_14

    processed = np.delete(processed, (4), axis=1)  # rsi_change
    processed = np.delete(processed, (3), axis=1)  # mid_point_diff

    processed = np.delete(processed, (2), axis=1)  # volume_over_volume_sma_50

    processed = np.delete(processed, (1), axis=1)  # open_close

    processed = np.delete(processed, (0), axis=1)  # close_close

    max_abs_scaler_close = MaxAbsScaler(copy=False)
    first_elements = processed[:, 0]
    first_elements = first_elements.reshape(-1, 1)
    max_abs_scaler_close.fit_transform(first_elements)
    max_abs_scaler_processed = MaxAbsScaler(copy=False)
    processed = max_abs_scaler_processed.fit_transform(processed)
    max_abs_scaler_processed_closed = MaxAbsScaler(copy=False)
    processed_close_10 = max_abs_scaler_processed_closed.fit_transform(processed_close_10)

    train_data_x, train_data_y = sliding_windows_look_ahead(processed, sequence_length, processed_close_10, 10)
    train_size = int(len(train_data_x) * percent_data_for_training)

    x_train_np = train_data_x[0:train_size]
    x_eval_np = train_data_x[train_size:len(train_data_x)]
    y_train_np = train_data_y[0:train_size]

    if percent_data_for_training == 1:
        x_eval_np = None
        dump(max_abs_scaler_processed, open('saved_models/data_scaler_biweekly.pkl', 'wb'))
        dump(max_abs_scaler_close, open('saved_models/close_scaler_biweekly.pkl', 'wb'))

    return x_train_np, x_eval_np, y_train_np, df


def load_training_data_monthly(sequence_length, percent_data_for_training, data_path):
    (df, processed, processed_close_5, processed_close_10, processed_close_20) = get_processed_data(data_path)

    # df['Close Log Dif'] = processed[:, 0]

    # Delete unwanted data points from processed
    processed = np.delete(processed, (11), axis=1)  # HA_close
    processed = np.delete(processed, (10), axis=1)  # HA_low
    processed = np.delete(processed, (9), axis=1)  # HA_high
    processed = np.delete(processed, (8), axis=1)  # HA_open
    # processed = np.delete(processed, (7), axis=1)  # MACDh_12_26_9
    # processed = np.delete(processed, (6), axis=1)  # sma_20_over_50
    # processed = np.delete(processed, (5), axis=1)  # rsi_14

    processed = np.delete(processed, (4), axis=1)  # rsi_change
    processed = np.delete(processed, (3), axis=1)  # mid_point_diff

    # processed = np.delete(processed, (2), axis=1)  # volume_over_volume_sma_50

    processed = np.delete(processed, (1), axis=1)  # open_close

    # processed = np.delete(processed, (0), axis=1)  # close_close

    max_abs_scaler_close = MaxAbsScaler(copy=False)
    first_elements = processed[:, 0]
    first_elements = first_elements.reshape(-1, 1)
    max_abs_scaler_close.fit_transform(first_elements)
    max_abs_scaler_processed = MaxAbsScaler(copy=False)
    processed = max_abs_scaler_processed.fit_transform(processed)
    max_abs_scaler_processed_closed = MaxAbsScaler(copy=False)
    processed_close_20 = max_abs_scaler_processed_closed.fit_transform(processed_close_20)

    train_data_x, train_data_y = sliding_windows_look_ahead(processed, sequence_length, processed_close_20, 20)
    train_size = int(len(train_data_x) * percent_data_for_training)

    x_train_np = train_data_x[0:train_size]
    x_eval_np = train_data_x[train_size:len(train_data_x)]
    y_train_np = train_data_y[0:train_size]

    if percent_data_for_training == 1:
        x_eval_np = None
        dump(max_abs_scaler_processed, open('saved_models/data_scaler_biweekly.pkl', 'wb'))
        dump(max_abs_scaler_close, open('saved_models/close_scaler_biweekly.pkl', 'wb'))

    return x_train_np, x_eval_np, y_train_np, df
