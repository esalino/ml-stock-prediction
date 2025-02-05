import prediction.model as m
import data.data_utils as data_utils
import torch
import numpy as np
import random as random
import os as os
from pickle import load


def predict():
    # Set device so can use gpu's if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Pytorch can give inconsistent results if you dont seed the
    # different libraries with the same seed
    seed_everything()

    # configure parameters
    sequence_length = 40
    hidden_dim = 60
    output_dim = 1
    num_layers = 1
    model_path = "./saved_models/pytorch_stocks_weekly_model.pth"
    data_path = "./raw_data/SPY.csv"

    eval_data_np, eval_raw_np, scaler = data_utils.load_prediction_data_weekly(sequence_length, data_path)
    eval_data = torch.from_numpy(eval_data_np).type(torch.Tensor)

    input_dim = eval_data_np.shape[2]

    model = m.create_model(input_dim, hidden_dim, output_dim, num_layers)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))

    with torch.no_grad():
        model.eval()
        data_predict = model(eval_data)

    scalar_close = load(open('./saved_models/close_scaler_weekly.pkl', 'rb'))
    data_predict = data_predict.data.numpy()
    data_predict = scalar_close.inverse_transform(data_predict)

    print(data_predict)


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    predict()
