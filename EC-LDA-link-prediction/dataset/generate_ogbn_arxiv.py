from torch_geometric.datasets import Reddit2
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch_geometric.transforms as T
import torch
from dataset.utils.dataset_utils import split_communities,split_dataset