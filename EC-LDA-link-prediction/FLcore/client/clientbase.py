import copy
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.utils import negative_sampling
from FLcore.optimizer.dp_optimizer import DPSGD
from utils import perturb_adj_laplace
import torch_geometric.transforms as T
from torch_geometric.data import Data

class Client():
    def __init__(self,args,id,dataset):
        
        self.model = copy.deepcopy(args.model)
        self.args=args
        self.id = id
        self.dataset = dataset
        self.train_data,self.val_data,self.test_data = dataset
        self.test_data = self.test_data.to(self.args.device)
        self.num_nodes = self.train_data.num_nodes
        
        if self.args.defense:
            
            self.ori_train_data_dege_index = copy.deepcopy(self.train_data.edge_index)
            self.train_data.edge_index = perturb_adj_laplace(self.train_data.edge_index,args.epsilon)

            # count=0
            # for i in range(len(self.train_data.edge_index[0])):
            #     src = self.train_data.edge_index[0][i].item()
            #     end = self.train_data.edge_index[1][i].item()
            #     for j in range(len(self.ori_train_data_dege_index[0])):
            #         if (src == self.ori_train_data_dege_index[0][j].item() and end == self.ori_train_data_dege_index[1][j].item())\
            #         or (src == self.ori_train_data_dege_index[1][j].item() and end == self.ori_train_data_dege_index[0][j].item()):
            #             count+=1
            #             break
            # print(count)

            # graph = Data(x = self.train_data.x, y = self.train_data.y,edge_index = self.train_data.edge_index)
            # transform = T.RandomLinkSplit(num_val=0.0,num_test=0.0,is_undirected=False,add_negative_train_samples=False,neg_sampling_ratio=0.0)
            # train,val,test = transform(graph)
            # # sorted_graph,_ = torch.sort(graph.edge_index, dim=1)
            # # sorted_train,__ = torch.sort(train.edge_index, dim=1)
            # # 此时的 flag 为 True，说明graph与train一致
            # # flag = torch.equal(sorted_graph, sorted_train)
            # # a=0
            # self.train_data = train
        # self.num_edges = self.train_data.num_edges

        # if self.args.defense:
            # self.train_data.edge_index = perturb_adj_laplace(self.train_data.edge_index,args.epsilon)

        # self.batch_size = 128

        self.train_edge_label_index,self.train_edge_label = self.get_train_test_link(self.train_data,'train')
        self.train_edge_label_index = self.train_edge_label_index.to(self.args.device)
        self.train_edge_label = self.train_edge_label.to(self.args.device)
        if self.args.defense:
            self.train_positive_links = self.train_data.edge_index.size(1)
            self.train_negative_links = int(self.train_positive_links*args.neg_times)
        else:
            self.train_positive_links = self.train_data.edge_label_index.size(1)
            self.train_negative_links = int(self.train_data.edge_label_index.size(1)*args.neg_times)
        self.train_links_num = self.train_positive_links + self.train_negative_links

        self.test_edge_label_index,self.test_edge_label = self.get_train_test_link(self.test_data,'test')
        self.test_edge_label_index = self.test_edge_label_index.to(self.args.device)
        self.test_edge_label = self.test_edge_label.to(self.args.device)
        self.test_positive_links = self.test_data.edge_label_index.size(1)
        self.test_negative_links = int(self.test_data.edge_label_index.size(1)*args.neg_times)
        self.test_links_num = self.test_positive_links + self.test_negative_links

        self.learning_rate = args.learning_rate
        self.num_classes = args.num_classes
        self.local_epochs = args.local_epochs
        self.num_clients = args.num_clients
        
        
        
        # self.num_edges = self.train_data.num_edges

        self.loss = nn.CrossEntropyLoss()
        # if self.args.defense:
            # batch_size = self.num_nodes
            # # learning_rate = 0.5
            # # numEpoch = 15000
            # sigma = 1.23
            # momentum = 0.0
            # delta = 10 ** (-5)
            # max_norm = 0.1
            # self.optimizer = DPSGD(
            #     l2_norm_clip=max_norm,  # 裁剪范数
            #     noise_multiplier=sigma,
            #     minibatch_size=batch_size,  # 几个样本梯度进行一次梯度下降
            #     microbatch_size=1,  # 几个样本梯度进行一次裁剪，这里选择逐样本裁剪
            #     params=self.model.parameters(),
            #     lr=self.args.learning_rate_dp,
            #     momentum=momentum
            # )

        
        if self.args.dataset in ['Cora','FacebookPagePage','CiteSeer','WikiCS','CoraFull','LastFM']:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,weight_decay=1e-4)
        elif self.args.dataset in ['PubMed'] :
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,weight_decay=1e-4)

        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay

    def get_train_test_link(self,data,flag):
        # if flag == 'train':
        #     scale = self.args.neg_times
        # else:
        #     scale = 1
        if self.args.defense and flag == 'train':
            neg_edge_index = negative_sampling(
            edge_index=data.edge_index, num_nodes=data.num_nodes,
            num_neg_samples=int(data.edge_index.size(1)*self.args.neg_times), method='sparse')
    
            edge_label_index = torch.cat(
                [data.edge_index, neg_edge_index],
                dim=-1,
            )
            edge_label = torch.cat([
                # data.edge_label,
                data.edge_label.new_ones(data.edge_index.size(1)),
                data.edge_label.new_zeros(neg_edge_index.size(1))
            ], dim=0)
        else:
            neg_edge_index = negative_sampling(
                edge_index=data.edge_index, num_nodes=data.num_nodes,
                num_neg_samples=int(data.edge_label_index.size(1)*self.args.neg_times), method='sparse')
        
            edge_label_index = torch.cat(
                [data.edge_label_index, neg_edge_index],
                dim=-1,
            )

            edge_label = torch.cat([
                data.edge_label,
                data.edge_label.new_zeros(neg_edge_index.size(1))
            ], dim=0)
        edge_label = edge_label.long()

        return edge_label_index,edge_label

    def set_parameters(self, model):  # 覆盖model.parameters()的操作；是get/set这种类型的操作
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()  # 深拷贝

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone() 


    def test_metrics(self):
        self.model.eval()  # 设置成“测试模式”,简单理解成不用反向传播了

        test_acc = 0
        test_num = 0   
        # for params in self.model.parameters():
        #     print(params.data)
            
        with torch.no_grad():
            out = self.model(self.test_data.x,self.test_data.edge_index,self.test_edge_label_index)
            pred = out.argmax(dim=1)
            test_correct = torch.sum(pred==self.test_edge_label).item()
            # test_correct = pred[self.dataset.test_mask] == self.dataset.y[self.dataset.test_mask]
            # test_acc = int(test_correct.sum()) / int(self.dataset.test_mask.sum())
            return int(test_correct),int(self.test_links_num)
    
    def train_metrics(self):
        self.model.eval()
        train_num = 0
        losses = 0
        with torch.no_grad():
            out = self.model(self.train_data.x,self.train_data.edge_index,self.train_edge_label_index)
            loss = self.loss(out,self.train_edge_label)
            pred = out.argmax(dim=1)
            train_correct = torch.sum(pred==self.train_edge_label).item()
            train_num += self.train_links_num
            losses += loss.item()*self.train_links_num
        
        return losses,train_num,train_correct