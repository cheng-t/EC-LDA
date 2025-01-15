from torch_geometric.datasets import Reddit2
import torch_geometric.transforms as T
import torch
from dataset.utils.dataset_utils import split_communities,split_dataset,process_dataset

def generate_Reddit2(num_clients,split,rand_seed):

    dataset = Reddit2(root='data/Reddit',transform=T.LargestConnectedComponents())

    return process_dataset(dataset,num_clients,split,rand_seed)

