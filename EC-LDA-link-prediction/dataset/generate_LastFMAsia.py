from torch_geometric.datasets import LastFMAsia
import torch_geometric.transforms as T
import torch
from dataset.utils.dataset_utils import split_communities,split_dataset,process_dataset

def generate_LatsFMAsia(num_clients,split,rand_seed):

    dataset = LastFMAsia(root='data/LastFMAsia', transform=T.LargestConnectedComponents())

    return process_dataset(dataset,num_clients,split,rand_seed)
    

