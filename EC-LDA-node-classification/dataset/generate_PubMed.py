from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch
from dataset.utils.dataset_utils import split_communities,split_dataset

def generate_PubMed(num_clients,split,rand_seed):

    dataset = Planetoid(root='data/Planetoid', name='PubMed', transform=T.LargestConnectedComponents())

    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]  # Get the graph object.

    print(data)
    print('==============================================================')

    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')

    client_data =  split_communities(data,num_clients,rand_seed)

    for k in range(len(client_data)):
        client_data[k]=split_dataset(client_data[k], split)

    return client_data,dataset.num_classes



