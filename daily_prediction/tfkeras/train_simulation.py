import daily_prediction.tfkeras.model as m
import data.data_utils as data_utils
import numpy as np
import random
import os
import utils.utils as utils


def train_simulation():

    seed_everything()

    # configure parameters
    num_epochs = 30
    sequence_length = 50
    hidden_dim = 50
    output_dim = 1
    percent_data_for_training = 0.95
    start_date = "1993-02-01"
    # data_path = "./raw_data/S&P500Daily.csv"
    data_path = "./raw_data/SPY.csv"

    x_train_np, x_eval_np, y_train_np, dataframe = data_utils.load_training_data_v2(
        sequence_length,
        percent_data_for_training,
        data_path)

    # x_train_np, x_eval_np, y_train_np, normalized_full_data_set, data_scaler, scalar_close = data_utils.load_training_data(
    #     sequence_length,
    #     percent_data_for_training,
    #     data_path,
    #     "1993-02-01")

    input_dim = x_train_np.shape[2]

    model = m.create_model(hidden_dim, output_dim, sequence_length, input_dim)

    model.fit(x_train_np, y_train_np, epochs=num_epochs)
    training_prediction = model.predict(x_eval_np)

    # Evaluate based on prediction if the next days prediction was in the correct direction
    # i.e. did it predict correctly if tomorrow was an up day or down day
    utils.evaluate_daily_simulation_v2(training_prediction, x_eval_np, dataframe)
    #utils.evaluate_daily_simulation(normalized_full_data_set, data_scaler, training_prediction, scalar_close, x_eval_np)


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    train_simulation()
