import torch

class SCoNeLayer(torch.nn.Module):
    # see https://arxiv.org/pdf/2102.10058: Algorithm 1 and Algorithm S-2
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W0 = torch.nn.Linear(in_features, out_features, bias=False)
        self.W1 = torch.nn.Linear(in_features, out_features, bias=False)
        self.W2 = torch.nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, B1, B2):
        # d2 = B2 @ B2^T @ x @ W2^l
        d2 = torch.matmul(B2, torch.matmul(B2.t(), self.W2(x)))
        
        # d1 = x @ W1^l
        d1 = self.W1(x)
        
        # d0 = B1^T @ B1 @ x @ W0^l
        d0 = torch.matmul(B1.t(), torch.matmul(B1, self.W0(x)))
        
        return torch.tanh(d2 + d1 + d0)

class SCoNe(torch.nn.Module):
    """
    SCoNe (Simplicial Complex Net) module for trajectory prediction on graphs.

    Args:
        in_features (int): Number of input features.
        hidden_features (int): Number of hidden features.
        num_layers (int): Number of layers in the SCoNe module.
    """
    def __init__(self, in_features, hidden_features, num_layers):
        super().__init__()
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList([
            SCoNeLayer(in_features if i == 0 else hidden_features, hidden_features)
            for i in range(num_layers)
        ])
        self.W0_L = torch.nn.Linear(hidden_features, 1, bias=False)

    def forward(self, x, B1, B2):
        """
        Forward pass of the SCoNe module.

        Args:
            x (torch.Tensor): Input tensor.
            B1 (torch.Tensor): Graph boundary matrix (C_1 -> C_0).
            B2 (torch.Tensor): Graph boundary matrix (C_2 -> C_1).

        Returns:
            torch.Tensor: Output tensor.
        """
        for layer in self.layers:
            x = layer(x, B1, B2)

        # x = B1 @ x @ W0^L
        x = torch.matmul(B1, self.W0_L(x))
        return x