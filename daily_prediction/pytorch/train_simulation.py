import daily_prediction.pytorch.model as m
import data.data_utils as data_utils
import torch
import numpy as np
import random as random
import os as os
import utils.utils as utils


def train():
    # Set device so can use gpu's if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Pytorch can give inconsistent results if you dont seed the
    # different libraries with the same seed
    seed_everything()

    # configure parameters
    # num_epochs = 30  # 50 best so far
    # sequence_length = 50
    # hidden_dim = 30
    num_epochs = 30
    sequence_length = 50
    hidden_dim = 60
    output_dim = 1
    num_layers = 1
    percent_data_for_training = 0.95
    #data_path = "./raw_data/S&P500Daily.csv"
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

    # Get as tensors
    x_train = torch.from_numpy(x_train_np).type(torch.Tensor)
    y_train = torch.from_numpy(y_train_np).type(torch.Tensor)
    x_eval = torch.from_numpy(x_eval_np).type(torch.Tensor)

    # Set device
    x_train, y_train = x_train.to(device), y_train.to(device)

    # Create model and setup loss function and optimiser
    model = m.create_model(input_dim, hidden_dim, output_dim, num_layers)
    model = model.to(device)
    loss_fn = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train model
    for t in range(num_epochs):
        # Zero out gradient, else they will accumulate between epochs
        optimiser.zero_grad()

        # Forward pass
        y_prediction = model(x_train)

        # Loss pass
        loss = loss_fn(y_prediction, y_train)

        # Backward pass
        loss.backward()

        # Update parameters
        optimiser.step()

        print("Epoch ", t, "MSE: ", loss.item())

    # Evaluate
    with torch.no_grad():
        model.eval()
        train_predict = model(x_eval)

    training_prediction = train_predict.data.numpy()

    # Evaluate based on prediction if the next days prediction was in the correct direction
    # i.e. did it predict correctly if tomorrow was an up day or down day
    utils.evaluate_daily_simulation_v2(training_prediction, x_eval_np, dataframe)
    #utils.evaluate_daily_simulation(normalized_full_data_set, data_scaler, training_prediction, scalar_close, x_eval_np)


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    train()
