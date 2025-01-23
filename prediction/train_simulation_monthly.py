import data.data_utils as data_utils
import utils.utils as utils
import prediction.model_utils as model_utils
import torch


def train_monthly():
    # configure parameters
    num_epochs = 40
    sequence_length = 60
    hidden_dim = 70
    percent_data_for_training = .85

    data_path = "./raw_data/SPY.csv"

    x_train_np, x_eval_np, y_train_np, dataframe = data_utils.load_training_data_monthly(
        sequence_length,
        percent_data_for_training,
        data_path)

    model = model_utils.train(num_epochs, hidden_dim, x_train_np, y_train_np)

    if (percent_data_for_training < 1):
        training_prediction = model_utils.run_prediction(model, x_eval_np)
        utils.evaluate_simulation_vbt(training_prediction, x_eval_np, dataframe, 20)
    else:
        torch.save(model.state_dict(), "./saved_models/pytorch_stocks_monthly_model.pth")


if __name__ == '__main__':
    train_monthly()
