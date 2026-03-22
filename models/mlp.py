import torch
import torch.nn.functional as F

class MLP(torch.nn.Module):
    """
    Multi-Layer Perceptron (MLP) model.

    Args:
        input_dim (int): The dimensionality of the input features.
        hidden_dims (int or list): The dimensions of the hidden layers. If an int is provided, a single hidden layer
            with that dimension will be created. If a list is provided, multiple hidden layers with the specified
            dimensions will be created.
        activation (function, optional): The activation function to be applied to the hidden layers. Defaults to
            the ReLU activation function.
    """
    def __init__(self, input_dim, hidden_dims, activation=F.relu):
        super().__init__()
        if type(hidden_dims) == int:
            hidden_dims = [hidden_dims]
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.layers = []

        last_dim = input_dim
        for dim in hidden_dims:
            self.layers.append(torch.nn.Linear(last_dim, dim))
            last_dim = dim
        self.layers.append(torch.nn.Linear(last_dim, 1))

        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        for layer in self.layers:
            x = layer(x)
            if layer != self.layers[-1]:  # Apply activation except for the last layer
                x = self.activation(x)
        return x
