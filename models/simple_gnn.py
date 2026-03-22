import torch
import torch_geometric
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class SimpleGNN(torch.nn.Module):
	"""
	Simple Graph Neural Network (GNN) model.

	Args:
		node_features (int): Number of input node features.
		gcn_dims (list): List of dimensions for the GCN layers.
		fc_dims (list): List of dimensions for the fully connected layers.
		gcn_activation (torch.nn.Module): Activation function for the GCN layers. Defaults to F.relu.
		fc_activation (torch.nn.Module): Activation function for the fully connected layers. Defaults to F.relu.
	"""

	def __init__(self, node_features, gcn_dims, fc_dims, gcn_activation=F.relu, fc_activation=F.relu):
		super().__init__()
		self.gcn_dims = gcn_dims
		self.fc_dims = fc_dims
		self.gcn_activation = gcn_activation
		self.fc_activation = fc_activation
		self.gcn_layers = []

		last_layer_dim = node_features
		for d in gcn_dims:
			self.gcn_layers.append(GCNConv(last_layer_dim, d))
			last_layer_dim = d

		self.linear_layers = []
		for d in fc_dims:
			self.linear_layers.append(torch.nn.Linear(last_layer_dim, d))
			last_layer_dim = d
		# append prediction layer
		self.linear_layers.append(torch.nn.Linear(last_layer_dim, 1))

		self.gcn_layers = torch.nn.ModuleList(self.gcn_layers)
		self.linear_layers = torch.nn.ModuleList(self.linear_layers)

	def forward(self, x, edge_index):
		"""
		Forward pass of the SimpleGNN model.

		Args:
			x (torch.Tensor): Input node features.
			edge_index (torch.Tensor): Graph edge indices.

		Returns:
			torch.Tensor: Output tensor.
		"""
		for layer in self.gcn_layers:
			x = layer(x, edge_index)
			x = self.gcn_activation(x)

		for i, layer in enumerate(self.linear_layers):
			x = layer(x)
			if i < len(self.linear_layers) - 1:
				x = self.fc_activation(x)

		return x