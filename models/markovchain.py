import numpy as np
import scipy.sparse as sp
from bidict import bidict
from sklearn.preprocessing import normalize
import torch
import random


class TrajectoryMarkovChain():
    """
    Markov chain model for trajectory prediction.

    Args:
        order (int): The order of the Markov chain.
        data (list): A list of trajectories.
    """
    def __init__(self, order, data):
        self.order = order
        if self.order < 1:
            raise ValueError(f'Invalid markov chain order {self.order}')
               
        paths = []
        for traj in data:
            if len(traj) >= self.order:
                for i in range(len(traj) - self.order):
                    paths.append(traj[i:i + self.order + 1])
 
        history_set, target_set = [], []
        for h in paths:
            history_set.append(frozenset(h[:-1].cpu().numpy()))
            target_set.append(h[-1].item())        
        history_set = set(history_set)
        target_set = set(target_set)

        self.history_to_idx = bidict({history : idx for idx, history in enumerate(history_set)})
        self.target_to_idx = bidict({history : idx for idx, history in enumerate(target_set)})
        i, j = [], []
        for history in paths:
            j.append(self.history_to_idx[frozenset(history[:-1].cpu().numpy())])
            i.append(self.target_to_idx[history[-1].item()])
        self.transition_counts = sp.coo_matrix((np.ones(len(paths), dtype=float), (i, j)), 
                                               shape=(len(self.target_to_idx), len(self.history_to_idx)))
        self.num_states = len(self.target_to_idx)

        self.transition_matrix = normalize(self.transition_counts.tocsc(), norm='l1', axis=0)
        
    
    def predict(self, history, data, device):
        """
        Predicts the next node in the trajectory given a history.

        Args:
            history (torch.Tensor): The history of nodes in the trajectory.
            data (Graph): The graph data.
            device (torch.device): The device to perform computations on.

        Returns:
            torch.Tensor: The predicted next node in the trajectory.
            bool: A flag indicating whether the prediction is random or not.
        """
        if frozenset(history.cpu().numpy()) not in self.history_to_idx:   
            neighbors = list(data.graph.get_neighbors(history[-1].item()))
            endpoint = random.choice(neighbors)
            random_flag = True
            return torch.tensor(endpoint, device=device), random_flag
        
        next_node_idx = torch.from_numpy(self.transition_matrix[:, self.history_to_idx[frozenset(history.cpu().numpy())]].todense()).to(device).view(-1)
        endpoints = torch.tensor([self.target_to_idx.inverse[idx] for idx in range(self.num_states)], device=device)
        endpoint_idx = np.random.choice(a=len(next_node_idx), size= 1, p= next_node_idx.cpu())
        endpoint = endpoints[endpoint_idx].squeeze()
        random_flag = False
        return endpoint, random_flag