import data.data_utils as data_utils
import utils.utils as utils
import prediction.model_utils as model_utils
import torch


def train_biweekly():
    # configure parameters
    num_epochs = 40
    sequence_length = 50
    hidden_dim = 70
    percent_data_for_training = .80

    data_path = "./raw_data/SPY_2024_12_06.csv"
    # data_path = "./raw_data/SPY.csv"

    x_train_np, x_eval_np, y_train_np, dataframe = data_utils.load_training_data_biweekly(
        sequence_length,
        percent_data_for_training,
        data_path)

    model = model_utils.train(num_epochs, hidden_dim, x_train_np, y_train_np)

    if (percent_data_for_training < 1):
        training_prediction = model_utils.run_prediction(model, x_eval_np)
        utils.evaluate_simulation_vbt(training_prediction, x_eval_np, dataframe, 10)
    else:
        torch.save(model.state_dict(), "./saved_models/pytorch_stocks_biweekly_model.pth")


if __name__ == '__main__':
    train_biweekly()
