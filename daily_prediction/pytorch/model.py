import torch
import torch.nn as nn


class GRU(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim=1,
                 num_layers=1):
        super(GRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define the GRU layer
        self.gru = nn.GRU(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, input):
        # Initialize cell state with zeros
        hn = torch.zeros(self.num_layers, input.size(0), self.hidden_dim)

        out, hn = self.gru(input, hn)

        # pass the linear layer the hidden state of the last index
        # can get this with out[:, -1, :] or hn.view(-1, self.hidden_dim)
        out = self.linear(out[:, -1, :])

        return out


def create_model(input_dim, hidden_dim, output_dim, num_layers):
    return GRU(input_dim, hidden_dim, output_dim=output_dim, num_layers=num_layers)
