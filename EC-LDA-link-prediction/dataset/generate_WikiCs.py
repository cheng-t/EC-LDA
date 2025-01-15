from torch_geometric.datasets import WikiCS
import torch_geometric.transforms as T
import torch
from dataset.utils.dataset_utils import split_communities,split_dataset,process_dataset

def generate_WikiCS(num_clients,split,rand_seed):

    dataset = WikiCS(root='data/WikiCS',transform=T.LargestConnectedComponents(),is_undirected=False)
   

    return process_dataset(dataset,num_clients,split,rand_seed)

