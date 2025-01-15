from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch
from dataset.utils.dataset_utils import split_communities,split_dataset,process_dataset

def generate_PubMed(num_clients,split,rand_seed):

    dataset = Planetoid(root='data/Planetoid', name='PubMed', transform=T.LargestConnectedComponents())

    return process_dataset(dataset,num_clients,split,rand_seed)



