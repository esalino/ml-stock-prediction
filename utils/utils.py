import math
import numpy as np
from datetime import datetime
import calendar


def evaluate_daily_simulation_v3(training_prediction, x_eval_np, dataframe):
    batch_test_size = x_eval_np.shape[0]

    # get the close list for compare. Because of _y, _x_eval_np ends one day before
    # end of main list which is why we are getting one day extra.
    dates = dataframe['Date']
    dates = dates[len(dates) - batch_test_size - 1:].tolist()
    raw_close = dataframe['Close']
    raw_close = raw_close[len(raw_close) - batch_test_size - 1:].tolist()
    raw_open = dataframe['Open']
    raw_open = raw_open[len(raw_open) - batch_test_size - 1:].tolist()

    training_prediction = np.reshape(training_prediction, batch_test_size)

    count_correct_direction = 0

    # Simulate trading at open of each dy. We will also short down
    # day predictions.
    tomorrow_is_long = training_prediction[0] >= 0
    start_price = raw_open[1]
    start_cash = 10000
    cash = start_cash
    current_shares = math.floor(cash / start_price)
    current_invested = current_shares * start_price
    cash = cash - current_invested if tomorrow_is_long else cash + current_invested
    is_long = tomorrow_is_long

    buy_hold_shares = math.floor(start_cash / raw_open[1])
    buy_hold_invested = buy_hold_shares * raw_open[1]
    buy_hold_cash = start_cash - buy_hold_invested
    buy_hold_cash += (buy_hold_shares * raw_close[len(raw_open) - 1])

    for i in range(2, len(raw_close) - 1):
        raw_open_today = raw_open[i]
        actual_direction = "equal"
        if raw_close[i - 1] > raw_close[i - 2]:
            actual_direction = "up"
        elif raw_close[i - 1] < raw_close[i - 2]:
            actual_direction = "down"

        prediction_direction = "equal"
        if training_prediction[i - 2] > 0:
            prediction_direction = "up"
        elif training_prediction[i - 2] < 0:
            prediction_direction = "down"

        is_correct = actual_direction == prediction_direction or actual_direction == "equal"
        if is_correct:
            count_correct_direction += 1

        prediction = training_prediction[i - 1]
        if prediction > 0:
            if not is_long:
                # buy back short position
                cash -= (current_shares * raw_open_today)
                # buy new long position
                current_shares = math.floor(cash / raw_open_today)
                current_invested = current_shares * raw_open_today
                cash -= current_invested
                is_long = True
        elif prediction < 0:
            if is_long:
                # sell long position
                cash += (current_shares * raw_open_today)
                # open short position
                current_shares = math.floor(cash / raw_open_today)
                cash += (current_shares * raw_open_today)
                is_long = False

    if is_long:
        cash += (current_shares * raw_open[len(raw_close) - 1])
    else:
        cash -= (current_shares * raw_open[len(raw_close) - 1])
    current_shares = 0

    print("Buy and Hold Final Cash: ", buy_hold_cash)
    print("Long Return: ", ((raw_close[len(raw_close) - 1] - start_price) / start_price) * 100)
    print("Weekly Model Trading Final Cash: ", cash)
    print("Weekly Model Return: ", ((cash - start_cash) / start_cash) * 100)
    print("Total Days Predicted: ", len(training_prediction))
    print("Days Predicted Correctly: ", count_correct_direction)
    print("Predicted Rate: ", (count_correct_direction / len(training_prediction)) * 100)


def evaluate_daily_simulation(training_prediction, x_eval_np, dataframe):
    batch_test_size = x_eval_np.shape[0]

    dates = dataframe['Date']
    dates = dates[len(dates) - batch_test_size:].tolist()
    raw_close = dataframe['Close']
    raw_close = raw_close[len(raw_close) - batch_test_size:].tolist()
    raw_open = dataframe['Open']
    raw_open = raw_open[len(raw_open) - batch_test_size:].tolist()

    training_prediction = np.reshape(training_prediction, batch_test_size)

    count_correct_direction = 0

    for i in range(1, len(training_prediction)):
        actual_direction = "equal"
        if raw_close[i] > raw_close[i - 1]:
            actual_direction = "up"
        elif raw_close[i] < raw_close[i - 1]:
            actual_direction = "down"

        prediction_direction = "equal"
        if training_prediction[i - 1] > 0:
            prediction_direction = "up"
        elif training_prediction[i - 1] < 0:
            prediction_direction = "down"

        is_correct = actual_direction == prediction_direction or actual_direction == "equal"
        if is_correct:
            count_correct_direction += 1

    print("Total Days Predicted: ", len(training_prediction - 1))
    print("Days Predicted Correctly: ", count_correct_direction)
    print("Predicted Rate: ", (count_correct_direction / (len(training_prediction) - 1)) * 100)


def getIndexForAllStartDaysWeekly(dates):
    indexes = []
    first_index = getIndexForFirstTradingDay(dates)
    i = first_index
    while i < len(dates):
        date_obj = datetime.strptime(dates[i], '%Y-%m-%d')
        day_of_week_number = calendar.weekday(date_obj.year, date_obj.month, date_obj.day)
        start_week_index = getIndexForNextStartOfWeek(i, day_of_week_number, dates)
        if start_week_index == -1:
            break
        indexes.append(start_week_index)
        i = start_week_index
    return indexes


def getIndexForFirstTradingDay(dates):
    for i in range(len(dates)):
        date_obj = datetime.strptime(dates[i], '%Y-%m-%d')
        day_of_week_number = calendar.weekday(date_obj.year, date_obj.month, date_obj.day)
        if day_of_week_number == 0:
            return i
    return -1


def getIndexForNextEndOfWeek(start_index, start_day_of_week_number, dates):
    current_day_of_week_number = start_day_of_week_number
    for i in range(start_index + 1, len(dates)):
        date_obj = datetime.strptime(dates[i], '%Y-%m-%d')
        day_of_week_number = calendar.weekday(date_obj.year, date_obj.month, date_obj.day)
        if day_of_week_number > current_day_of_week_number:
            current_day_of_week_number = day_of_week_number
        else:
            return i - 1
    return -1


def getIndexForNextStartOfWeek(start_index, start_day_of_week_number, dates):
    current_day_of_week_number = start_day_of_week_number
    for i in range(start_index + 1, len(dates)):
        date_obj = datetime.strptime(dates[i], '%Y-%m-%d')
        day_of_week_number = calendar.weekday(date_obj.year, date_obj.month, date_obj.day)
        if day_of_week_number > current_day_of_week_number:
            current_day_of_week_number = day_of_week_number
        else:
            return i
    return -1


def evaluate_weekly_simulation_v3(training_prediction, x_eval_np, dataframe):
    batch_test_size = x_eval_np.shape[0]

    # get the close list for compare. We need previous one dayof predictions.
    dates = dataframe['Date']
    dates = dates[len(dates) - batch_test_size - 1:].tolist()
    raw_close = dataframe['Close']
    raw_close = raw_close[len(raw_close) - batch_test_size - 1:].tolist()
    raw_open = dataframe['Open']
    raw_open = raw_open[len(raw_open) - batch_test_size - 1:].tolist()

    training_prediction = np.reshape(training_prediction, batch_test_size)

    count_correct_direction = 0
    first_days = getIndexForAllStartDaysWeekly(dates)

    # Simulate trading at open of each dy. We will also short down
    # day predictions.
    tomorrow_is_long = training_prediction[first_days[0] - 1] >= 0
    start_price = raw_open[first_days[0]]
    start_cash = 10000
    cash = start_cash
    current_shares = math.floor(cash / start_price)
    current_invested = current_shares * start_price
    cash = cash - current_invested if tomorrow_is_long else cash + current_invested
    is_long = tomorrow_is_long

    buy_hold_shares = math.floor(start_cash / raw_open[1])
    buy_hold_invested = buy_hold_shares * raw_open[1]
    buy_hold_cash = start_cash - buy_hold_invested
    buy_hold_cash += (buy_hold_shares * raw_close[len(raw_open) - 1])

    for i in range(1, len(first_days)):
        current_index = first_days[i]
        previous_index = first_days[i - 1]
        raw_open_today = raw_open[current_index]
        actual_direction = "equal"
        if raw_close[current_index - 1] > raw_close[previous_index - 1]:
            actual_direction = "up"
        elif raw_close[current_index - 1] < raw_close[previous_index - 1]:
            actual_direction = "down"

        prediction_direction = "equal"
        if training_prediction[current_index - 1] > 0:
            prediction_direction = "up"
        elif training_prediction[current_index - 1] < 0:
            prediction_direction = "down"

        is_correct = actual_direction == prediction_direction or actual_direction == "equal"
        if is_correct:
            count_correct_direction += 1

        prediction = training_prediction[first_days[i] - 1]
        if prediction > 0:
            if not is_long:
                # buy back short position
                cash -= (current_shares * raw_open_today)
                # buy new long position
                current_shares = math.floor(cash / raw_open_today)
                current_invested = current_shares * raw_open_today
                cash -= current_invested
                is_long = True
        elif prediction < 0:
            if is_long:
                # sell long position
                cash += (current_shares * raw_open_today)
                # open short position
                current_shares = math.floor(cash / raw_open_today)
                cash += (current_shares * raw_open_today)
                is_long = False

    if is_long:
        cash += (current_shares * raw_open[len(raw_close) - 1])
    else:
        cash -= (current_shares * raw_open[len(raw_close) - 1])
    current_shares = 0

    print("Buy and Hold Final Cash: ", buy_hold_cash)
    print("Long Return: ", ((raw_close[len(raw_close) - 1] - start_price) / start_price) * 100)
    print("Weekly Model Trading Final Cash: ", cash)
    print("Weekly Model Return: ", ((cash - start_cash) / start_cash) * 100)
    print("Total Days Predicted: ", len(first_days))
    print("Days Predicted Correctly: ", count_correct_direction)
    print("Predicted Rate: ", (count_correct_direction / len(first_days)) * 100)


def evaluate_weekly_simulation_v4(training_prediction, x_eval_np, dataframe):
    batch_test_size = x_eval_np.shape[0]

    # get the close list for compare. We need previous one dayof predictions.
    dates = dataframe['Date']
    dates = dates[len(dates) - batch_test_size:].tolist()
    raw_close = dataframe['Close']
    raw_close = raw_close[len(raw_close) - batch_test_size:].tolist()
    raw_open = dataframe['Open']
    raw_open = raw_open[len(raw_open) - batch_test_size:].tolist()

    training_prediction = np.reshape(training_prediction, batch_test_size)

    count_correct_direction = 0

    for i in range(5, len(training_prediction)):
        actual_direction = "equal"
        if raw_close[i] > raw_close[i - 5]:
            actual_direction = "up"
        elif raw_close[i] < raw_close[i - 5]:
            actual_direction = "down"

        prediction_direction = "equal"
        if training_prediction[i - 5] > 0:
            prediction_direction = "up"
        elif training_prediction[i - 5] < 0:
            prediction_direction = "down"

        is_correct = actual_direction == prediction_direction or actual_direction == "equal"
        if is_correct:
            count_correct_direction += 1

    print("Total Days Predicted: ", len(training_prediction) - 5)
    print("Days Predicted Correctly: ", count_correct_direction)
    print("Predicted Rate: ", (count_correct_direction / (len(training_prediction) - 5)) * 100)


def evaluate_weekly_simulation(training_prediction, x_eval_np, dataframe):
    batch_test_size = x_eval_np.shape[0]

    # get the close list for compare. We need previous one dayof predictions.
    dates = dataframe['Date']
    dates = dates[len(dates) - batch_test_size - 1:].tolist()
    start_index = getIndexForFirstTradingDay(dates)
    raw_open = dataframe['Open']
    raw_open = raw_open[len(raw_open) - batch_test_size - 1:].tolist()
    raw_close = dataframe['Close']
    raw_close = raw_close[len(raw_close) - batch_test_size - 1:].tolist()

    training_prediction = np.reshape(training_prediction, batch_test_size)

    # Simulate trading at start of new week. We will also short down
    # weekly predictions.
    start_cash = 10000
    cash = start_cash
    is_long = False

    buy_hold_shares = math.floor(start_cash / raw_open[start_index])
    buy_hold_invested = buy_hold_shares * raw_open[start_index]
    buy_hold_cash = start_cash - buy_hold_invested
    buy_hold_cash += (buy_hold_shares * raw_open[len(raw_open) - 1])

    count_correct_direction = 0
    current_shares = 0

    i = start_index
    while i < len(raw_open):
        date_obj = datetime.strptime(dates[i], '%Y-%m-%d')
        day_of_week_number = calendar.weekday(date_obj.year, date_obj.month, date_obj.day)
        end_week_index = getIndexForNextEndOfWeek(i, day_of_week_number, dates)
        if end_week_index == -1:
            break

        actual_direction = "equal"
        if raw_close[i] > raw_close[i - 1]:
            actual_direction = "up"
        elif raw_close[i] < raw_close[i - 1]:
            actual_direction = "down"

        prediction_direction = "equal"
        if training_prediction[i - 2] > 0:
            prediction_direction = "up"
        elif training_prediction[i - 2] < 0:
            prediction_direction = "down"

        is_correct = actual_direction == prediction_direction or actual_direction == "equal"
        if is_correct:
            count_correct_direction += 1

        prediction = training_prediction[i - 1]
        if prediction > 0:
            if not is_long:
                current_shares = math.floor(cash / raw_open[i])
                current_invested = current_shares * raw_open[i]
                cash -= current_invested
                is_long = True
        elif prediction < 0:
            if is_long:
                cash += (current_shares * raw_open[i])
                current_shares = 0
                is_long = False

        i = end_week_index + 1

    if is_long:
        cash += (current_shares * raw_open[len(raw_open) - 1])
        current_shares = 0
        is_long = False

    print("Buy and Hold Final Cash: ", buy_hold_cash)
    print("Weekly Model Final Cash: ", cash)
