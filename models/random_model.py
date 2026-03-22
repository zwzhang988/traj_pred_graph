import torch

class RandomModel(torch.nn.Module):
    """
    A random model that generates random predictions.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Forward pass of the random model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The (random) output tensor.
        """
        result = torch.rand((x.shape[0], 1), device=x.device)
        return result