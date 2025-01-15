from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx
import torch
import numpy as np
import torch_geometric.transforms as T

def split_communities(data,num_client,rand_seed):
    
    G = to_networkx(data, to_undirected=True, node_attrs=['x','y'])
    
    communities = sorted(nx.community.asyn_fluidc(G, num_client, max_iter = 100, seed= rand_seed))
    # communities = sorted(nx.community.asyn_fluidc(G, num_client, max_iter = 50, seed= 0))

    node_groups = []
    for com in communities:
        node_groups.append(list(com))   
    list_of_clients = []

    for i in range(num_client):
        list_of_clients.append(from_networkx(G.subgraph(node_groups[i]).copy()))

    return list_of_clients

def split_dataset(data,split_percentage):
    transform = T.RandomLinkSplit(num_val=0.05,num_test=1-split_percentage,is_undirected=True,add_negative_train_samples=False,neg_sampling_ratio=0.0)
    train,val,test = transform(data)
    if test.edge_label_index.size(1)==0:
        print("no enough test link data!")
        exit()
    return [train,val,test]
    # mask = torch.randn((data.num_nodes)) < split_percentage
    # nmask = torch.logical_not(mask)

    # train_mask = mask
    # test_mask = nmask
    # data.train_mask = train_mask
    # data.test_mask = test_mask
    # if test_mask.numel()==0:
    #     print('no enough test data!')
    #     exit()
    # return data


def process_dataset(dataset,num_clients,split,rand_seed):

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