import torch
import numpy as np



def calculate_hidden_init_limits(layer):
    """
    Calculate the initialization limits for the hidden layer.

    Args:
        layer (torch.nn.Module): The hidden layer for which to calculate initialization limits.

    Returns:
        tuple: A tuple containing the lower and upper limits for the initialization.
    """
    # Get the number of input units (fan-in) for the given hidden layer
    fan_in = layer.weight.data.size()[0]

    # Calculate the initialization limit using the 1/sqrt(fan_in) formula
    # This is a common initialization technique to improve convergence during training.
    lim = 1. / np.sqrt(fan_in)

    # The initialization limits are symmetric around zero, so we return (-lim, lim)
    # This means the initial weights of the hidden layer will be randomly
    # initialized within the range [-lim, lim].
    return (-lim, lim)

class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, upper_bound, activation_fun=torch.nn.Tanh(), output_activation=None, batch_norm=False):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.output_activation = output_activation
        layer_sizes = [self.input_size] + self.hidden_sizes
        self.layers = torch.nn.ModuleList([ torch.nn.Linear(i, o) for i,o in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.activations = [activation_fun for l in  self.layers]
        self.readout = torch.nn.Linear(self.hidden_sizes[-1], self.output_size)

        self.batch_norm = batch_norm
        self.b0 = torch.nn.BatchNorm1d(self.input_size)
        self.batch_norms = [torch.nn.BatchNorm1d(self.hidden_sizes[i]) for i in range(len(self.layers))]
        self.upper_bound = upper_bound

        self.init_weights()

    # Initialize the weights of the hidden layers
    def init_weights(self):
        for layer in self.layers:
            layer.weight.data.uniform_(*calculate_hidden_init_limits(layer))
            #layer.bias.data.uniform_(*calculate_hidden_init_limits(layer))

    def forward(self, x, type):

        x = self.b0(x) if self.batch_norm else x

        if type == "actor":
            if x.dim() == 1:
                print("X dim is 1")
                x = x.unsqueeze(0)

            if self.batch_norm is True:
                for layer,activation_fun, batch_norm in zip(self.layers, self.activations, self.batch_norms):
                    x = activation_fun(batch_norm(layer(x)))
            else:
                for layer,activation_fun in zip(self.layers, self.activations):
                    x = activation_fun(layer(x))

        elif type == "critic":

            for layer, activation_fun in zip(self.layers, self.activations):
                x = activation_fun(layer(x))

        if self.output_activation is not None:
            if self.upper_bound is not None:
                return self.output_activation(self.readout(x)) * self.upper_bound
            else:
                return self.output_activation(self.readout(x))
        else:
            return self.readout(x)


    def predict(self, x):

        with torch.no_grad():
            try:
                x = torch.from_numpy(x.astype(np.float32)).numpy()
            except:
                x = x

            for layer, activation_fun in zip(self.layers, self.activations):
                x = activation_fun(layer(x))
            if self.output_activation is not None:
                if self.upper_bound is not None:
                    return self.output_activation(self.readout(x)) * self.upper_bound
                else:
                    return self.output_activation(self.readout(x))
            else:
                return self.readout(x)
