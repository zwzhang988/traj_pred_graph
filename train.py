import torch
import torch_geometric
import h5py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.nn.dense import DenseGCNConv
import torch.nn.functional as F
from tqdm import tqdm
from argparse import ArgumentParser
from time import time
import wandb
import random
import os
from datetime import datetime

from graph import TrajectoryEncoder, Trajectory
from datasets import GeoLifeTrajectoryDataset, TDriveTrajectoryDataset, PneumaTrajectoryDataset, Distance_evaluation
from models.simple_gnn import SimpleGNN
import models.egnn as eg
import models.mlp as mlp
from models.scone import SCoNe
from models.gat import GATNetwork
from models.random_model import RandomModel
from evaluation import AutoregressiveEvaluation
import models.markovchain as mc

parser = ArgumentParser()
parser.add_argument('--model', default=None, type=str, help='Type of model to train (required)')
parser.add_argument('--dataset', default=None, type=str, help='Choose dataset (required)')
parser.add_argument('--n_samples', default=-1, type=int, help='Number of samples (default: all)')
parser.add_argument('--min_length', default=4, type=int, help='Minimum trajectory length (default: 4)')
parser.add_argument('--max_length', default=10000, type=int, help='Maximum trajectory length (default: 10,000)')
parser.add_argument('--min_history', default=3, type=int, help='Minimum history for model context (default: 3)')
parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility (default: 42)')
parser.add_argument('--no_test', default=False, action='store_true', help='Join validation and test set (default: False)')
parser.add_argument('--evaluate_test', default=10, type=int, help='Evaluate the test set autoregressively every x epochs')
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate (default: 1e-3)')
parser.add_argument('--epochs', default=10, type=int, help="Number of epochs for training (default: 10)")
parser.add_argument('--batch_size', default=32, type=int, help="Batch size (default: 32)")
parser.add_argument('--patience', default=5, type=int, help="Patience for early stopping (default: 5)")
parser.add_argument('--trajectory_encoding', default='linear', help="Type of trajectory encoding ('uniform', 'linear', 'transformer') default: 'linear'")
parser.add_argument('--sweep', default=False, action='store_true', help='WandB sweep (default: False)')

def predict(model, model_type, batch, data, device=None):
    # prediction
    if model_type == 'simple_gnn' or model_type == 'gat':
        on_path = traject_encoder.encode_trajectory(batch)
        nfeatures = torch.concat([data.graph.node_coordinates, data.graph.nodes, on_path], dim=1).to(dtype=torch.float32)

        next_node = model(nfeatures, data.graph.edge_index)
    elif model_type == 'egnn':
        on_path = traject_encoder.encode_trajectory(batch)
        nfeatures = torch.concat([data.graph.nodes, on_path], dim=1).to(dtype=torch.float32)

        next_node, _ = model(nfeatures, data.graph.node_coordinates, edges=data.graph.edge_index, edge_attr=None)
    elif model_type == 'mlp':
        on_path = traject_encoder.encode_trajectory(batch)
        nfeatures = torch.concat([data.graph.node_coordinates, data.graph.nodes, on_path], dim=1).to(dtype=torch.float32)

        next_node = model(nfeatures)
    elif model_type == 'scone':
        nfeatures = Trajectory.trajectory_to_chain(batch, data.graph, device=device).unsqueeze(dim=-1)

        next_node = model(nfeatures, data.graph.B1, data.graph.B2)
    elif args.model == "random":
        next_node = model(on_path)
    
    return next_node

def evaluate_autoregressive(args, model, data, coordinates, dataloader, min_history, traject_encoder, auto_eval, device=None):
    """
    Evaluate the model autoregressively on the given data.

    Args:
        args: The commandline arguments.
        model: The model to evaluate.
        data: The data used for evaluation.
        coordinates: The coordinates of the nodes in the graph.
        dataloader: The dataloader for iterating over the evaluation data.
        min_history: The minimum history length for autoregressive prediction.
        traject_encoder: The trajectory encoder for encoding trajectories.
        auto_eval: The evaluation object for computing metrics.
        device: The device to use for computation (default: None).

    Returns:
        metrics: A dictionary containing the evaluation metrics.
    """
    metrics = {
        "accuracy": {
            "avg": 0.0,
            "final": 0.0
        },
        "euclid_distance": {
            "avg": 0.0,
            "final": 0.0
        },
        "path_hops": {
            "avg": 0.0,
            "final": 0.0
        },
        "path_length": {
            "avg": 0.0,
            "final": 0.0
        },
        "random_ratio": 0
    }
    n_samples = 0
    random_count = 0
    pred_sum = 0
    for batch in dataloader:
        batch = batch.squeeze(0)

        # autoregressive evaluation
        predictions = []
        targets = batch[min_history:]
        path = batch[:min_history]
        
        for t in targets:
            if args.model == "markovchain":
                if min_history < model.order: raise ValueError(f'Invalid min_history {min_history}')
                history = path[-model.order:]
                next_node, random_flag = model.predict(history, data, device)
                predictions.append(next_node)
                path = torch.cat([path, predictions[-1].unsqueeze(dim=0)], dim=0)
                pred_sum += 1
                if random_flag: random_count += 1
            
            else:
                with torch.no_grad():
                    next_node = predict(model, args.model, path, data, device)
                
                # p for all nodes >= 0
                next_node = F.softmax(next_node, dim=0)
        
                last_node = path[-1].item()
                last_node_neighbors = list(data.graph.get_neighbors(last_node))
                neighbor_mask = torch.zeros(next_node.shape[0], 1, device=device)
                neighbor_mask[last_node_neighbors] = 1
            
                predictions.append((neighbor_mask * next_node).argmax())
                path = torch.cat([path, predictions[-1].unsqueeze(dim=0)], dim=0)

        acc = auto_eval.eval_accuracy(predictions, targets)
        euc = auto_eval.eval_euclidean_distance(data.graph, predictions, targets)
        path_metrics = auto_eval.eval_shortest_path(data.graph, predictions, targets)

        metrics['accuracy']['avg'] += acc['average_accuracy']
        metrics['accuracy']['final'] += acc['final_accuracy']

        metrics['euclid_distance']['avg'] += euc['average_distance']
        metrics['euclid_distance']['final'] += euc['final_distance']

        metrics['path_hops']['avg'] += path_metrics['average_hops']
        metrics['path_hops']['final'] += path_metrics['final_hops']

        metrics['path_length']['avg'] += path_metrics['average_path_length']
        metrics['path_length']['final'] += path_metrics['final_path_length']

        if args.model == 'markovchain':
            metrics['random_ratio'] = random_count / pred_sum
        else:
            metrics['random_ratio'] = 0.0

        n_samples += 1

    # nomalize metrics
    for k1 in metrics.keys():
        if k1 == 'random_ratio': continue
        for k2 in metrics[k1].keys():
            metrics[k1][k2] /= n_samples
    
    # update look-up dict file
    auto_eval.update_stored_dict()
    
    return metrics

def main(args):
    """
    Main method of training.

    Args:
        args: The commandline arguments.
    """
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Validate commandline arguments
    if args.model is None:
        print("No model specified. Use '--model <model_name>'")
        exit(-1)
    if args.dataset is None:
        print("No dataset specified. Use '--dataset <dataset_name>'")
        exit(-1)
    if args.min_history >= args.min_length:
        print("--min_length has to be greater than --min_history")
        exit(-1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")
    
    dataset_kwargs = {
        'n_samples': args.n_samples,
        'min_trajectory_length': args.min_length,
        'max_trajectory_length': args.max_length,
        'device': device
    }
    testset_kwargs = {
        'n_samples': -1, # always load full testset
        'min_trajectory_length': args.min_length,
        'max_trajectory_length': args.max_length,
        'device': device
    }
    PATH_PREFIX = "/ceph/hdd/students/weea/trajectory-prediction-on-graphs/datasets/"
    if args.dataset == "GeoLife" or args.dataset == "geolife":
        data = GeoLifeTrajectoryDataset(PATH_PREFIX + "geolife_train.h5", **dataset_kwargs)
        test_data = GeoLifeTrajectoryDataset(PATH_PREFIX + "geolife_test.h5", **testset_kwargs)
    elif args.dataset == "TDrive" or args.dataset == "tdrive":
        data = TDriveTrajectoryDataset(PATH_PREFIX + "tdrive_train.h5", **dataset_kwargs)
        test_data = TDriveTrajectoryDataset(PATH_PREFIX + "tdrive_test.h5", **testset_kwargs)
    elif args.dataset == "Pneuma" or args.dataset == "pneuma":
        data = PneumaTrajectoryDataset(PATH_PREFIX + "pneuma_train.h5", **dataset_kwargs)
        test_data = PneumaTrajectoryDataset(PATH_PREFIX + "pneuma_test.h5", **testset_kwargs)
    else:
        raise NotImplementedError("Unknown dataset")
    
    print(f"Using dataset with {len(data)} samples in total")

    auto_eval = AutoregressiveEvaluation(args.dataset.lower(), data.graph.node_coordinates, data.graph.edges)

    gen = torch.Generator().manual_seed(args.seed)
    train_data, val_data = torch.utils.data.random_split(data, [0.8, 0.2], generator=gen)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=1)
    if args.no_test:
        testloader = torch.utils.data.DataLoader(val_data, batch_size=1)
    else:
        testloader = torch.utils.data.DataLoader(test_data, batch_size=1)
    
    if args.model == "markovchain":
        model = mc.TrajectoryMarkovChain(order= 2, data= data)
        wandb_run = wandb.init(
            project="mllab",
            name= "Markovchain_pneuma_order_4",
            config={
                "model": args.model,
                "dataset": args.dataset,
                "n_samples": args.n_samples,
                "len_data": len(data),
                "device": device,
                "min_length": args.min_length,
                "max_length": args.max_length,
                "min_history": args.min_history,
                "seed": args.seed,
                "no_test": args.no_test,
            }
        )
        start_time = time()
        min_history = args.min_history
        
        auto_metrics = evaluate_autoregressive(args, model, data, coordinates=None, dataloader=testloader, min_history=min_history, traject_encoder=None, auto_eval=auto_eval, device=device)

        print(f"Time auto validation: {time() - start_time:.2f}s")

        wandb.log({            
            "test/accuracy_avg": auto_metrics['accuracy']['avg'],
            "test/accuracy_final": auto_metrics['accuracy']['final'],
            "test/euclidean_avg": auto_metrics['euclid_distance']['avg'],
            "test/euclidean_final": auto_metrics['euclid_distance']['final'],
            "test/path_hops_avg": auto_metrics['path_hops']['avg'],
            "test/path_hops_final": auto_metrics['path_hops']['final'],
            "test/path_length_avg": auto_metrics['path_length']['avg'],
            "test/path_length_final": auto_metrics['path_length']['final'],
            "test/random_ratio": auto_metrics['random_ratio']
        })
    else:
        if args.sweep:
            run = wandb.init(project="mllab")
            LR = wandb.config.lr
            BATCH_SIZE = wandb.config.batch_size
            TRAJECTORY_ENCODING = wandb.config.trajectory_encoding
            GAT_DIMS = wandb.config.gat_dims
            FC_DIMS = wandb.config.fc_dims
            N_HEADS = wandb.config.n_heads
            HIDDEN_FEATURES = wandb.config.hidden_features
            NUM_LAYERS = wandb.config.num_layers

        else:
            wandb_run = wandb.init(
                project="mllab",
                
                # track hyperparameters and run metadata
                config={
                    "model": args.model,
                    "dataset": args.dataset,
                    "n_samples": args.n_samples,
                    "len_data": len(data),
                    "device": device,
                    "min_length": args.min_length,
                    "max_length": args.max_length,
                    "min_history": args.min_history,
                    "seed": args.seed,
                    "no_test": args.no_test,
                    "lr": args.lr,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "trajectory_encoding": args.trajectory_encoding
                }
            )
            LR = args.lr
            BATCH_SIZE = args.batch_size
            TRAJECTORY_ENCODING = args.trajectory_encoding
            # example hparameters
            HIDDEN_FEATURES = 16
            NUM_LAYERS = 3
            GCN_DIMS = [16, 16]
            GAT_DIMS = [16, 16]
            FC_DIMS = [16, 16]
            N_HEADS = 4

        trajectory_dim = 1
        if TRAJECTORY_ENCODING == 'transformer':
            trajectory_dim = 2

        if args.model == "simple_gnn":
            # Use node_coordinates (2dim) also as input feature
            model = SimpleGNN(2 + data.graph.nodes.shape[1] + trajectory_dim, gcn_dims=GCN_DIMS, fc_dims=FC_DIMS)
            path = '../best_model/' + args.model + '_' + args.dataset + '_' + str(args.min_history) + '_' + str(args.lr)  + '_' + str(args.batch_size) + '_' + args.trajectory_encoding + '_' + \
                str(model.gcn_dims) + '_'  + str(model.fc_dims) + '_best_model.pth'
                
        elif args.model == "egnn":
            model = eg.EGNN(in_node_nf=data.graph.nodes.shape[1] + trajectory_dim, hidden_nf=32, out_node_nf=1, in_edge_nf=1, device=device)
            path = '../best_model/' + args.model + '_' + args.dataset + '_' + str(args.min_history) + '_' + str(args.lr)  + '_' + str(args.batch_size) + '_' + args.trajectory_encoding + '_' + \
                str(model.hidden_nf) + '_' + str(model.n_layers) + '_best_model.pth'
        elif args.model == "scone":
            model = SCoNe(in_features=1, hidden_features=HIDDEN_FEATURES, num_layers=NUM_LAYERS)
            data.graph.B1 = data.graph.B1.to(device)
            data.graph.B2 = data.graph.B2.to(device)
            path = '../best_model/' + args.model + '_' + args.dataset + '_' + str(args.min_history) + '_' + str(args.lr)  + '_' + str(args.batch_size) + '_' + args.trajectory_encoding + '_' + \
                str(model.hidden_features) + '_' + str(model.num_layers) + '_best_model.pth'
        elif args.model == "gat":
            model = GATNetwork(node_features=2 + data.graph.nodes.shape[1] + trajectory_dim, gat_dims=GAT_DIMS, fc_dims=FC_DIMS, n_heads=N_HEADS)
            path = '../best_model/' + args.model + '_' + args.dataset + '_' + str(args.min_history) + '_' + str(args.lr)  + '_' + str(args.batch_size) + '_' + args.trajectory_encoding + '_' + \
                str(model.gat_dims) + '_' + str(model.fc_dims) + '_' + str(model.n_heads) + '_best_model.pth'
        elif args.model == 'mlp':
            # input dimension: coordinate_dim + node_dim + trajectory_dim
            input_dim = 2 + data.graph.nodes.shape[1] + trajectory_dim
            model = mlp.MLP(input_dim=input_dim, hidden_dims=HIDDEN_FEATURES)
            path = '../best_model/' + args.model + '_' + args.dataset + '_' + str(args.min_history) + '_' + str(args.lr)  + '_' + str(args.batch_size) + '_' + args.trajectory_encoding + '_' + \
                str(model.hidden_dims) + '_' + str(model.activation) + '_best_model.pth'
        elif args.model == "random":
            model = RandomModel()
        else:
            raise NotImplementedError("Unknown model")
        
        wandb.config["path"] = path

        loss_f = torch.nn.CrossEntropyLoss()
        if args.model != "random":
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
        model.to(device)
  
        traject_encoder = TrajectoryEncoder(data.graph.node_coordinates.shape[0], TRAJECTORY_ENCODING, 2, device)
        min_history = args.min_history
        patience = args.patience
        if args.model == "random":
            epochs = []
        else:
            epochs = tqdm(range(args.epochs), desc='')
        tlosses = []
        taccuracies = []
        vlosses = []
        vaccuracies = []

        # Early stopping
        best_model_stats = {}
        cur_patience = 0
        val_random_lengths = []
        for epoch in epochs:
            wandb.log({"epoch": epoch+1})
            model.train()
            epoch_accuracy = []
            epoch_loss = []
            batch_counter = 0
            loss = 0
            start_time = time()
            for batch in trainloader:
                batch = batch.squeeze(0).to(device)

                # randomize length
                min_length = min_history +1
                max_length = batch.shape[0]
                rand_length = random.randint(min_length, max_length)
                batch = batch[:rand_length]

                next_node = predict(model, args.model, batch[:-1], data, device)
                
                last_node = batch[-2].item()
                y_node = batch[-1].item()
                
                last_node_neighbors = list(data.graph.get_neighbors(last_node))
                neighbor_scores = next_node[last_node_neighbors]

                target_node = torch.tensor(last_node_neighbors.index(y_node), device=device)
                loss += loss_f(neighbor_scores.squeeze(1), target_node)

                batch_counter += 1
                epoch_accuracy.append(torch.mean((neighbor_scores.argmax(dim=0) == last_node_neighbors.index(batch[-1].cpu().item()))*1.0).cpu().item())

                if batch_counter >= BATCH_SIZE:
                    loss /= batch_counter
                    loss.backward()
                    epoch_loss.append(loss.item())
                    optimizer.step()
                    optimizer.zero_grad()
                    wandb.log({
                        "train/loss": epoch_loss[-1]
                    })
                    batch_counter = 0
                    loss = 0
            # process left-over batch
            if batch_counter > 0:
                loss /= batch_counter
                loss.backward()
                epoch_loss.append(loss.item())
                optimizer.step()
                optimizer.zero_grad()
                wandb.log({
                    "train/loss": epoch_loss[-1]
                })

            taccuracies.append(np.array(epoch_accuracy).mean())
            tlosses.append(np.array(epoch_loss).mean())
            wandb.log({
                "train/accuracy": taccuracies[-1]
            })
            print(f"\nTime training: {time() - start_time:.2f}s")
            start_time = time()

            # Validation
            model.eval()
            epoch_accuracy = []
            epoch_loss = []
            for batch_ix, batch in enumerate(valloader):
                batch = batch.squeeze(0)

                if epoch == 0:
                    min_length = min_history +1
                    max_length = batch.shape[0]
                    rand_length = random.randint(min_length, max_length)
                    val_random_lengths.append(rand_length)
                else:
                    rand_length = val_random_lengths[batch_ix]
                    
                batch = batch[:rand_length]
                x = batch[:-1]
                y = batch[-1]

                with torch.no_grad():
                    next_node = predict(model, args.model, x, data, device)
                    
                    last_node = x[-1].item()
                    y_node = y.item()
                    
                    last_node_neighbors = list(data.graph.get_neighbors(last_node))
                    neighbor_scores = next_node[last_node_neighbors]

                    target_node = torch.tensor(last_node_neighbors.index(y_node), device=device)
                    loss = loss_f(neighbor_scores.squeeze(1), target_node)

                    epoch_accuracy.append(torch.mean((neighbor_scores.argmax(dim=0) == last_node_neighbors.index(y.cpu().item()))*1.0).cpu().item())
                    epoch_loss.append(loss.item())

            vaccuracies.append(np.array(epoch_accuracy).mean())
            vlosses.append(np.array(epoch_loss).mean())
            wandb.log({
                "val/loss": vlosses[-1],
                "val/accuracy": vaccuracies[-1]
            })

            print(f"Time validation: {time() - start_time:.2f}s")

            if (epoch + 1) % args.evaluate_test == 0:
                start_time = time()
                auto_metrics = evaluate_autoregressive(args, model, data, data.graph.node_coordinates, testloader, min_history, traject_encoder=traject_encoder, auto_eval=auto_eval, device=device)
                print(f"Time auto validation: {time() - start_time:.2f}s")

                wandb.log({            
                    "test/accuracy_avg": auto_metrics['accuracy']['avg'],
                    "test/accuracy_final": auto_metrics['accuracy']['final'],
                    "test/euclidean_avg": auto_metrics['euclid_distance']['avg'],
                    "test/euclidean_final": auto_metrics['euclid_distance']['final'],
                    "test/path_hops_avg": auto_metrics['path_hops']['avg'],
                    "test/path_hops_final": auto_metrics['path_hops']['final'],
                    "test/path_length_avg": auto_metrics['path_length']['avg'],
                    "test/path_length_final": auto_metrics['path_length']['final'],
                })
            
            
            if not best_model_stats or vlosses[-1] < best_model_stats['val_loss']:
                best_model_stats = {
                    "val_loss":vlosses[-1], 
                    "train_loss":tlosses[-1]
                }
                checkpoint = {
                    'epoch': epoch, 
                    'model_state_dict': model.state_dict(), 
                    'best_val_loss': vlosses[-1]
                }
                if args.model == 'simple_gnn':
                    checkpoint.update({'gcn_dims': model.gcn_dims})
                    checkpoint.update({'fc_dims': model.fc_dims})
                    checkpoint.update({'gcn_activation': model.gcn_activation})
                    checkpoint.update({'fc_activation': model.fc_activation})
                if args.model == 'gat':
                    checkpoint.update({'gat_dims': model.gat_dims})
                    checkpoint.update({'fc_dims': model.fc_dims})
                    checkpoint.update({'n_heads': model.n_heads})
                    checkpoint.update({'gat_activation': model.gat_activation})
                    checkpoint.update({'fc_activation': model.fc_activation})
                if args.model == 'egnn':
                    checkpoint.update({'hidden_nf': model.hidden_nf})
                    checkpoint.update({'n_layers': model.n_layers})
                if args.model == 'mlp':
                    checkpoint.update({'hidden_dims': model.hidden_dims})
                    checkpoint.update({'activation': model.activation})
                if args.model == 'scone':
                    checkpoint.update({'hidden_features': model.hidden_features})
                    checkpoint.update({'num_layers': model.num_layers})

                cur_patience = 0
            else:
                cur_patience += 1

            if patience and cur_patience >= patience:
                print("Stopping early at epoch {}!".format(epoch))           
                
                if not os.path.exists(path): 
                    torch.save(checkpoint, path)
                checkpoint_stored = torch.load(path)
                if checkpoint_stored['best_val_loss'] > checkpoint['best_val_loss']:
                    torch.save(checkpoint, path)
                
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                start_time = time()

                auto_metrics = evaluate_autoregressive(args, model, data, data.graph.node_coordinates, testloader, min_history, traject_encoder=traject_encoder, auto_eval=auto_eval, device=device)

                print(f"Time auto validation: {time() - start_time:.2f}s")

                wandb.log({            
                    "test/accuracy_avg": auto_metrics['accuracy']['avg'],
                    "test/accuracy_final": auto_metrics['accuracy']['final'],
                    "test/euclidean_avg": auto_metrics['euclid_distance']['avg'],
                    "test/euclidean_final": auto_metrics['euclid_distance']['final'],
                    "test/path_hops_avg": auto_metrics['path_hops']['avg'],
                    "test/path_hops_final": auto_metrics['path_hops']['final'],
                    "test/path_length_avg": auto_metrics['path_length']['avg'],
                    "test/path_length_final": auto_metrics['path_length']['final'],
                })
                break
            
            epochs.set_description(f"Train: {tlosses[-1]:.4f} {taccuracies[-1]:.3f}\tValidation: {vlosses[-1]:.4f} {vaccuracies[-1]:.3f}", refresh=False)
    
    # store model
    if args.model != 'markovchain':
        if args.model != "random":
            if not os.path.exists(path): 
                torch.save(checkpoint, path)
            checkpoint_stored = torch.load(path)
            if checkpoint_stored['best_val_loss'] > checkpoint['best_val_loss']:
                torch.save(checkpoint, path)
                    
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
        
        start_time = time()

        auto_metrics = evaluate_autoregressive(args, model, data, data.graph.node_coordinates, testloader, min_history, traject_encoder=traject_encoder, auto_eval=auto_eval, device=device)

        print(f"Time auto validation: {time() - start_time:.2f}s")

        wandb.log({            
            "test/accuracy_avg": auto_metrics['accuracy']['avg'],
            "test/accuracy_final": auto_metrics['accuracy']['final'],
            "test/euclidean_avg": auto_metrics['euclid_distance']['avg'],
            "test/euclidean_final": auto_metrics['euclid_distance']['final'],
            "test/path_hops_avg": auto_metrics['path_hops']['avg'],
            "test/path_hops_final": auto_metrics['path_hops']['final'],
            "test/path_length_avg": auto_metrics['path_length']['avg'],
            "test/path_length_final": auto_metrics['path_length']['final'],
        })
    print("Training finished")
    wandb.finish()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)