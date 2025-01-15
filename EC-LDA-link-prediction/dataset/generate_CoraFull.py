from torch_geometric.datasets import CoraFull
import torch_geometric.transforms as T
import torch
from dataset.utils.dataset_utils import split_communities,split_dataset,process_dataset

def generate_CoraFull(num_clients,split,rand_seed):

    # dataset = Planetoid(root='data/Planetoid', name='Cora', transform=T.LargestConnectedComponents())
    # transform=T.LargestConnectedComponents()保留图中的最大连通子图，即去除孤立节点
    dataset = CoraFull(root='data/CoraFull',transform=T.LargestConnectedComponents())
    # dataset = FacebookPagePage(root='data/FacebookPagePage')

    return process_dataset(dataset,num_clients,split,rand_seed)

