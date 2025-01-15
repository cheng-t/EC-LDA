from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch
from dataset.utils.dataset_utils import split_communities,split_dataset,process_dataset

def generate_CiteSeer(num_clients,split,rand_seed):
    transform = T.Compose([
    T.NormalizeFeatures(),
    T.LargestConnectedComponents()
    ])
    
    dataset = Planetoid(root='data/Planetoid', name='CiteSeer', transform=T.LargestConnectedComponents())

    return process_dataset(dataset,num_clients,split,rand_seed)


if __name__ == "__main__":
    generate_CiteSeer(10,0.8)
