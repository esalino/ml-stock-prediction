import prediction.model as m
import torch
import numpy as np
import random as random
import os as os


def train(num_epochs, hidden_dim, x_train_np, y_train_np):
    # Set device so can use gpu's if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Pytorch can give inconsistent results if you dont seed the
    # different libraries with the same seed
    seed_everything()

    # configure parameters
    output_dim = 1
    num_layers = 1

    input_dim = x_train_np.shape[2]

    # Get as tensors
    x_train = torch.from_numpy(x_train_np).type(torch.Tensor)
    y_train = torch.from_numpy(y_train_np).type(torch.Tensor)

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

    return model


def run_prediction(model, x_eval_np):
    x_eval = torch.from_numpy(x_eval_np).type(torch.Tensor)
    # Evaluate
    with torch.no_grad():
        model.eval()
        train_predict = model(x_eval)

    training_prediction = train_predict.data.numpy()

    return training_prediction


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    train()
