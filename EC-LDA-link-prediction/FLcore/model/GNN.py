from torch_geometric.nn import GCNConv, SAGEConv,GATConv
import torch.nn.functional as F
import torch
from torch import nn


class GCN(torch.nn.Module):
    # def __init__(self, hidden_channels1,hidden_channels2,hidden_channels3,features_in, features_out):
    def __init__(self,dataset,hops,features_in, c4096=4096,c1024=1024,c512=512,c256=256,c128=128,c64=64,c16=16):
        super().__init__()
        # self.conv1 = GCNConv(features_in, hidden_channels2)
        # self.conv2 = GCNConv(hidden_channels2, hidden_channels3)
        # self.conv3 = GCNConv(hidden_channels3, hidden_channels3)
        # # self.fc1 = nn.Linear(hidden_channels2,hidden_channels3)
        self.hops = hops
        # # self.activation = nn.LeakyReLU()
        # # self.activation = nn.ELU()
        if dataset in ['Cora','CiteSeer']:
            if hops == 1:
                self.conv1 = GCNConv(features_in,64)
            elif hops == 2:
                self.conv1 = GCNConv(features_in,512)
                self.conv2 = GCNConv(512,64)
            elif hops == 3:
                self.conv1 = GCNConv(features_in,512)
                self.conv2 = GCNConv(512,256)
                self.conv3 = GCNConv(256,64)
            else:
                print("Wrong hops!")
                exit()
        elif dataset in ['PubMed','FacebookPagePage','WikiCS']:
            if hops == 1:
                self.conv1 = GCNConv(features_in,64)
            elif hops == 2:
                self.conv1 = GCNConv(features_in,128)
                self.conv2 = GCNConv(128,64)
            elif hops == 3:
                self.conv1 = GCNConv(features_in,128)
                self.conv2 = GCNConv(128,128)
                self.conv3 = GCNConv(128,64)
            else:
                print("Wrong hops!")
                exit()
        elif dataset == 'CoraFull':
            if hops == 1:
                self.conv1 = GCNConv(features_in,64)
            elif hops == 2:
                self.conv1 = GCNConv(features_in,1024)
                self.conv2 = GCNConv(1024,64)
            elif hops == 3:
                self.conv1 = GCNConv(features_in,1024)
                self.conv2 = GCNConv(1024,128)
                self.conv3 = GCNConv(128,64)
            else:
                print("Wrong hops!")
                exit()
        elif dataset == 'LastFM':
            if hops == 1:
                self.conv1 = GCNConv(features_in,64)
            elif hops == 2:
                self.conv1 = GCNConv(features_in,128)
                self.conv2 = GCNConv(128,64)
            elif hops == 3:
                self.conv1 = GCNConv(features_in,128)
                self.conv2 = GCNConv(128,128)
                self.conv3 = GCNConv(128,64)
            else:
                print("Wrong hops!")
                exit()

        
        self.fc1 = nn.Linear(64,64)
        self.fc2 = nn.Linear(64,2)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index,edge_label_index):
        if self.hops == 1:
            x = self.conv1(x,edge_index)
            x = self.activation(x)
        elif self.hops == 2:
            x = self.conv1(x,edge_index)
            x = self.activation(x)
            x = self.conv2(x,edge_index)
            x = self.activation(x)
        else:
            x = self.conv1(x,edge_index)
            x = self.activation(x)
            x = self.conv2(x,edge_index)
            x = self.activation(x)
            x = self.conv3(x,edge_index)
            x = self.activation(x)

        # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.fc1(torch.cat((x[edge_label_index[0]],x[edge_label_index[1]]),dim=1))
        # x = self.fc1((x[edge_label_index[0]]*x[edge_label_index[1]]).sum(dim=-1).unsqueeze(1))

        # todo:只要一层，改代码
        # x = x[edge_label_index[0]]*x[edge_label_index[1]]
        x = self.fc1((x[edge_label_index[0]]*x[edge_label_index[1]]))
        x = self.activation(x)

        # return (x[edge_label_index[0]]*x[edge_label_index[1]]).sum(dim=-1)
        x = self.fc2(x)

        

        return x
    
class GAT(torch.nn.Module):
    # def __init__(self, hidden_channels1,hidden_channels2,hidden_channels3,features_in, features_out):
    #     super().__init__()
    #     self.conv1 = SAGEConv(features_in, hidden_channels2)
    #     self.conv2 = SAGEConv(hidden_channels2, hidden_channels3)
    #     # self.conv3 = GCNConv(hidden_channels3, hidden_channels3)
    #     self.fc1 = nn.Linear(hidden_channels3*2,hidden_channels3)
    #     self.fc2 = nn.Linear(hidden_channels3,2)

    # def forward(self, x, edge_index,edge_label_index):
    #     x = self.conv1(x, edge_index).relu()

    #     x = self.conv2(x, edge_index).relu()
    #     # x = self.conv3(x,edge_index)
    #     x = self.fc1(torch.cat((x[edge_label_index[0]],x[edge_label_index[1]]),dim=1)).relu()

    #     x = self.fc2(x)

    #     return x
    def __init__(self,dataset,hops,features_in, c4096=4096,c1024=1024,c512=512,c256=256,c128=128,c64=64,c16=16):
        super().__init__()
        # self.conv1 = GCNConv(features_in, hidden_channels2)
        # self.conv2 = GCNConv(hidden_channels2, hidden_channels3)
        # self.conv3 = GCNConv(hidden_channels3, hidden_channels3)
        # # self.fc1 = nn.Linear(hidden_channels2,hidden_channels3)
        self.hops = hops
        # # self.activation = nn.LeakyReLU()
        # # self.activation = nn.ELU()
        if dataset in ['Cora','CiteSeer']:
            if hops == 1:
                self.conv1 = GATConv(features_in,64)
            elif hops == 2:
                self.conv1 = GATConv(features_in,512)
                self.conv2 = GATConv(512,64)
            elif hops == 3:
                self.conv1 = GATConv(features_in,512)
                self.conv2 = GATConv(512,256)
                self.conv3 = GATConv(256,64)
            else:
                print("Wrong hops!")
                exit()
        elif dataset in ['PubMed','FacebookPagePage','WikiCS']:
            if hops == 1:
                self.conv1 = GATConv(features_in,64)
            elif hops == 2:
                self.conv1 = GATConv(features_in,128)
                self.conv2 = GATConv(128,64)
            elif hops == 3:
                self.conv1 = GATConv(features_in,128)
                self.conv2 = GATConv(128,128)
                self.conv3 = GATConv(128,64)
            else:
                print("Wrong hops!")
                exit()
        elif dataset == 'CoraFull':
            if hops == 1:
                self.conv1 = GATConv(features_in,64)
            elif hops == 2:
                self.conv1 = GATConv(features_in,1024)
                self.conv2 = GATConv(1024,64)
            elif hops == 3:
                self.conv1 = GATConv(features_in,1024)
                self.conv2 = GATConv(1024,128)
                self.conv3 = GATConv(128,64)
            else:
                print("Wrong hops!")
                exit()
        elif dataset == 'LastFM':
            if hops == 1:
                self.conv1 = GATConv(features_in,64)
            elif hops == 2:
                self.conv1 = GATConv(features_in,128)
                self.conv2 = GATConv(128,64)
            elif hops == 3:
                self.conv1 = GATConv(features_in,128)
                self.conv2 = GATConv(128,128)
                self.conv3 = GATConv(128,64)
            else:
                print("Wrong hops!")
                exit()

        
        self.fc1 = nn.Linear(64,64)
        self.fc2 = nn.Linear(64,2)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index,edge_label_index):
        if self.hops == 1:
            x = self.conv1(x,edge_index)
            x = self.activation(x)
        elif self.hops == 2:
            x = self.conv1(x,edge_index)
            x = self.activation(x)
            x = self.conv2(x,edge_index)
            x = self.activation(x)
        else:
            x = self.conv1(x,edge_index)
            x = self.activation(x)
            x = self.conv2(x,edge_index)
            x = self.activation(x)
            x = self.conv3(x,edge_index)
            x = self.activation(x)

        # x = F.dropout(x)
        
        # x = self.fc1(torch.cat((x[edge_label_index[0]],x[edge_label_index[1]]),dim=1))
        x = self.fc1((x[edge_label_index[0]]*x[edge_label_index[1]]))
        x = self.activation(x)

        x = self.fc2(x)

        return x

class GraphSAGE(torch.nn.Module):
    # def __init__(self, hidden_channels1,hidden_channels2,hidden_channels3,features_in, features_out):
    #     super().__init__()
    #     self.conv1 = GATConv(features_in, hidden_channels2)
    #     self.conv2 = GATConv(hidden_channels2, hidden_channels3)
    #     # self.conv3 = GCNConv(hidden_channels3, hidden_channels3)
    #     self.fc1 = nn.Linear(hidden_channels3*2,hidden_channels3)
    #     self.fc2 = nn.Linear(hidden_channels3,2)

    # def forward(self, x, edge_index,edge_label_index):
    #     x = self.conv1(x, edge_index).relu()

    #     x = self.conv2(x, edge_index).relu()
    #     # x = self.conv3(x,edge_index)
    #     x = self.fc1(torch.cat((x[edge_label_index[0]],x[edge_label_index[1]]),dim=1)).relu()

    #     x = self.fc2(x)

    #     return x
    def __init__(self,dataset,hops,features_in, c4096=4096,c1024=1024,c512=512,c256=256,c128=128,c64=64,c16=16):
        super().__init__()
        # self.conv1 = GCNConv(features_in, hidden_channels2)
        # self.conv2 = GCNConv(hidden_channels2, hidden_channels3)
        # self.conv3 = GCNConv(hidden_channels3, hidden_channels3)
        # # self.fc1 = nn.Linear(hidden_channels2,hidden_channels3)
        self.hops = hops
        # # self.activation = nn.LeakyReLU()
        # # self.activation = nn.ELU()
        if dataset in ['Cora','CiteSeer']:
            if hops == 1:
                self.conv1 = SAGEConv(features_in,64)
            elif hops == 2:
                self.conv1 = SAGEConv(features_in,512)
                self.conv2 = SAGEConv(512,64)
            elif hops == 3:
                self.conv1 = SAGEConv(features_in,512)
                self.conv2 = SAGEConv(512,256)
                self.conv3 = SAGEConv(256,64)
            else:
                print("Wrong hops!")
                exit()
        elif dataset in ['PubMed','FacebookPagePage','WikiCS']:
            if hops == 1:
                self.conv1 = SAGEConv(features_in,64)
            elif hops == 2:
                self.conv1 = SAGEConv(features_in,128)
                self.conv2 = SAGEConv(128,64)
            elif hops == 3:
                self.conv1 = SAGEConv(features_in,128)
                self.conv2 = SAGEConv(128,128)
                self.conv3 = SAGEConv(128,64)
            else:
                print("Wrong hops!")
                exit()
        elif dataset == 'CoraFull':
            if hops == 1:
                self.conv1 = SAGEConv(features_in,64)
            elif hops == 2:
                self.conv1 = SAGEConv(features_in,1024)
                self.conv2 = SAGEConv(1024,64)
            elif hops == 3:
                self.conv1 = SAGEConv(features_in,1024)
                self.conv2 = SAGEConv(1024,128)
                self.conv3 = SAGEConv(128,64)
            else:
                print("Wrong hops!")
                exit()
        elif dataset == 'LastFM':
            if hops == 1:
                self.conv1 = SAGEConv(features_in,64)
            elif hops == 2:
                self.conv1 = SAGEConv(features_in,128)
                self.conv2 = SAGEConv(128,64)
            elif hops == 3:
                self.conv1 = SAGEConv(features_in,128)
                self.conv2 = SAGEConv(128,128)
                self.conv3 = SAGEConv(128,64)
            else:
                print("Wrong hops!")
                exit()

        
        self.fc1 = nn.Linear(64,64)
        self.fc2 = nn.Linear(64,2)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index,edge_label_index):
        if self.hops == 1:
            x = self.conv1(x,edge_index)
            x = self.activation(x)
        elif self.hops == 2:
            x = self.conv1(x,edge_index)
            x = self.activation(x)
            x = self.conv2(x,edge_index)
            x = self.activation(x)
        else:
            x = self.conv1(x,edge_index)
            x = self.activation(x)
            x = self.conv2(x,edge_index)
            x = self.activation(x)
            x = self.conv3(x,edge_index)
            x = self.activation(x)

        # x = self.fc1(torch.cat((x[edge_label_index[0]],x[edge_label_index[1]]),dim=1))
        x = self.fc1((x[edge_label_index[0]]*x[edge_label_index[1]]))
        x = self.activation(x)

        x = self.fc2(x)

        return x