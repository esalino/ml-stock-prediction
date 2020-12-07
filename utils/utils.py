import math
import numpy as np


def evaluate_daily_simulation(normalized_full_data_set, data_scaler, training_prediction, scalar_close, x_eval_np):

    batch_test_size = x_eval_np.shape[0]
    train_predict_non_norm = scalar_close.inverse_transform(training_prediction)
    training_prediction = np.reshape(training_prediction, batch_test_size)
    train_predict_non_norm = np.reshape(train_predict_non_norm, batch_test_size)

    # Evaluate based on prediction if the next days prediction was in the correct direction
    # i.e. did it predict correctly if tomorrow was an up day or down day
    #close_normalized = normalized_full_data_set[:, 3]
    close_normalized = normalized_full_data_set[:,0]
    close_non_normalized = data_scaler.inverse_transform(normalized_full_data_set)
    #close_non_normalized = close_non_normalized[:, 3]
    close_non_normalized = close_non_normalized[:,0]
    count_correct_direction = 0
    start_price = close_non_normalized[len(close_normalized) - batch_test_size - 1]
    end_price =  close_non_normalized[len(close_non_normalized) - 1]
    start_cash = 10000
    cash = start_cash
    current_shares = math.floor(cash / start_price)
    current_invested = current_shares * start_price
    cash -= current_invested
    current_gains = 0
    is_long = close_normalized[len(close_normalized) - batch_test_size - 1] < training_prediction[0]

    count_actual_up = 0
    count_actual_down = 0
    count_actual_equal = 0
    predicted_correct_up = 0
    predicted_correct_down = 0
    predicted_correct_equal = 0
    predicted_up = 0
    predicted_down = 0
    predicted_equal = 0

    j = 0
    for i in range(len(close_normalized) - batch_test_size, len(close_normalized)):
        daily_gain_abs = 0
        #print("Previous Close: ", close_non_normalized[i - 1], " Current Close: ", close_non_normalized[i],
        #      " Predicted Close: ", train_predict_non_norm[j])

        actual_direction = "equal"
        if close_non_normalized[i] > close_non_normalized[i-1]:
            actual_direction = "up"
            count_actual_up += 1
        elif close_non_normalized[i] < close_non_normalized[i-1]:
            actual_direction = "down"
            count_actual_down += 1
        else:
            count_actual_equal += 1

        prediction_direction = "equal"
        if training_prediction[j] > close_normalized[i - 1]:
            prediction_direction = "up"
            predicted_up += 1
        elif training_prediction[j] < close_normalized[i - 1]:
            prediction_direction = "down"
            predicted_down += 1
        else:
            predicted_equal += 1

        is_correct = actual_direction == prediction_direction or actual_direction == "equal"
        if is_correct:
            count_correct_direction += 1
            daily_gain_abs += current_shares * math.fabs(close_non_normalized[i] - close_non_normalized[i - 1])
            if actual_direction == "up":
                predicted_correct_up += 1
            elif actual_direction == "down":
                predicted_correct_down += 1
            else:
                predicted_correct_equal += 1
        elif training_prediction[j] != close_normalized[i - 1]:
            daily_gain_abs -= (current_shares * math.fabs(close_non_normalized[i] - close_non_normalized[i - 1]))

        current_gains += daily_gain_abs
        tomorrow_is_long = tomorrow_is_long if j + 1 == len(training_prediction) else close_normalized[i] < training_prediction[
            j + 1]
        if tomorrow_is_long and not is_long:
            cash += (current_invested + current_gains)
            current_shares = math.floor(cash / close_non_normalized[i])
            current_invested = current_shares * close_non_normalized[i]
            cash -= current_invested
            is_long = True
            current_gains = 0
        elif not tomorrow_is_long and is_long:
            cash += (current_invested + current_gains)
            current_shares = math.floor(cash / close_non_normalized[i])
            current_invested = current_shares * close_non_normalized[i]
            cash -= current_invested
            is_long = False
            current_gains = 0
        if (j + 1) % 5 == 0 and (j + 1) != 0:
            print("total tested:", (j + 1), " total correct direction: ", count_correct_direction,
                  " correct prediction rate: ",
                  (count_correct_direction / (j + 1)) * 100, "%")
        j += 1

    print("total tested:", (j + 1), " total correct direction: ", count_correct_direction, " correct prediction rate: ",
          (count_correct_direction / (j + 1)) * 100, "%")

    cash += (current_invested + current_gains)
    total_return = ((cash - start_cash) / start_cash) * 100
    print("Long Return: ", str(((end_price - start_price) / start_price) * 100))
    print("Initial Investment: ", start_cash, ", Ending Cash: ", str(cash), ", Total Return: " + str(total_return))

    total_count = count_actual_up + count_actual_down + count_actual_equal
    print("actual up ", count_actual_up, "actual down ", count_actual_down, "actual equal ", count_actual_equal,
          "total ", total_count)
    print("actual up % ", (count_actual_up / total_count) * 100, "actual down % ",
          (count_actual_down / total_count) * 100, "actual equal % ", (count_actual_equal / total_count) * 100)
    total_predicted_correct = predicted_correct_up + predicted_correct_down + predicted_correct_equal
    print("predicted up ", predicted_correct_up, "predicted down ", predicted_correct_down, "predicted equal ",
          predicted_correct_equal,
          "total ", total_predicted_correct)
    print("(of actual) predicted up % ", (predicted_correct_up / count_actual_up) * 100, "predicted down % ",
          (predicted_correct_down / count_actual_down) * 100)
    print("(of predicted) predicted up % ", (predicted_correct_up / total_predicted_correct) * 100, "predicted down % ",
          (predicted_correct_down / total_predicted_correct) * 100)


def evaluate_daily_simulation_v2(training_prediction, x_eval_np, dataframe):
    batch_test_size = x_eval_np.shape[0]

    raw_close = dataframe['Close']
    raw_close = raw_close[len(raw_close) - batch_test_size - 1:].tolist()

    training_prediction = np.reshape(training_prediction, batch_test_size)

    count_correct_direction = 0
    count_actual_up = 0
    count_actual_down = 0
    count_actual_equal = 0
    predicted_correct_up = 0
    predicted_correct_down = 0
    predicted_correct_equal = 0
    predicted_up = 0
    predicted_down = 0
    predicted_equal = 0

    start_price = raw_close[len(raw_close) - batch_test_size - 1]
    end_price = raw_close[len(raw_close) - 1]
    start_cash = 10000
    cash = start_cash
    current_shares = math.floor(cash / start_price)
    current_invested = current_shares * start_price
    cash -= current_invested
    current_gains = 0
    is_long = training_prediction[0] >= 0
    tomorrow_is_long = is_long

    j = 0
    for i in range(1, len(raw_close)):
        daily_gain_abs = 0

        actual_direction = "equal"
        if raw_close[i] > raw_close[i - 1]:
            actual_direction = "up"
            count_actual_up += 1
        elif raw_close[i] < raw_close[i - 1]:
            actual_direction = "down"
            count_actual_down += 1
        else:
            count_actual_equal += 1

        prediction_direction = "equal"
        if training_prediction[j] > 0:
            prediction_direction = "up"
            predicted_up += 1
        elif training_prediction[j] < 0:
            prediction_direction = "down"
            predicted_down += 1
        else:
            predicted_equal += 1

        is_correct = actual_direction == prediction_direction or actual_direction == "equal"
        if is_correct:
            count_correct_direction += 1
            daily_gain_abs += current_shares * math.fabs(raw_close[i] - raw_close[i - 1])
            if actual_direction == "up":
                predicted_correct_up += 1
            elif actual_direction == "down":
                predicted_correct_down += 1
            else:
                predicted_correct_equal += 1
        else:
            daily_gain_abs -= current_shares * math.fabs(raw_close[i] - raw_close[i - 1])

        # print("daily_gain_abs ", daily_gain_abs)
        current_gains += daily_gain_abs
        tomorrow_is_long = tomorrow_is_long if j + 1 == len(training_prediction) else training_prediction[j + 1] >= 0

        if tomorrow_is_long and not is_long:
            cash += (current_invested + current_gains)
            current_shares = math.floor(cash / raw_close[i])
            current_invested = current_shares * raw_close[i]
            cash -= current_invested
            is_long = True
            current_gains = 0
        elif not tomorrow_is_long and is_long:
            cash += (current_invested + current_gains)
            current_shares = math.floor(cash / raw_close[i])
            current_invested = current_shares * raw_close[i]
            cash -= current_invested
            is_long = False
            current_gains = 0

        if (j + 1) % 5 == 0 and (j + 1) != 0:
            print("total tested:", (j + 1), " total correct direction: ", count_correct_direction,
                  " correct prediction rate: ",
                  (count_correct_direction / (j + 1)) * 100, "%")
        j += 1

    print("total tested:", (j + 1), " total correct direction: ", count_correct_direction, " correct prediction rate: ",
          (count_correct_direction / (j + 1)) * 100, "%")

    cash += (current_invested + current_gains)
    total_return = ((cash - start_cash) / start_cash) * 100
    print("Long Return: ", str(((end_price - start_price) / start_price) * 100))
    print("Initial Investment: ", start_cash, ", Ending Cash: ", str(cash), ", Total Return: " + str(total_return))

    total_count = count_actual_up + count_actual_down + count_actual_equal
    print("actual up ", count_actual_up, "actual down ", count_actual_down, "actual equal ", count_actual_equal,
          "total ", total_count)
    print("actual up % ", (count_actual_up / total_count) * 100, "actual down % ",
          (count_actual_down / total_count) * 100, "actual equal % ", (count_actual_equal / total_count) * 100)
    total_predicted_correct = predicted_correct_up + predicted_correct_down + predicted_correct_equal
    print("predicted up ", predicted_correct_up, "predicted down ", predicted_correct_down, "predicted equal ",
          predicted_correct_equal,
          "total ", total_predicted_correct)
    print("(of actual) predicted up % ", (predicted_correct_up / count_actual_up) * 100, "predicted down % ",
          (predicted_correct_down / count_actual_down) * 100)
    print("(of predicted) predicted up % ", (predicted_correct_up / total_predicted_correct) * 100, "predicted down % ",
          (predicted_correct_down / total_predicted_correct) * 100, "predicted equal % ",
          (predicted_correct_equal / total_predicted_correct) * 100)
