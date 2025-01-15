import torch.nn as nn
import numpy as np
import time
from FLcore.client.clientbase import Client
import sys
import copy
import random
from privacy_analysis.RDP.compute_rdp import compute_rdp
from privacy_analysis.RDP.rdp_convert_dp import compute_eps

class clientAVGblra(Client):
    def __init__(self,args,id,dataset):
        super().__init__(args,id,dataset)
        self.pre_model = copy.deepcopy(args.model)
        self.batch_gradient = None
        self.label_distribution = [0 for i in range(2)]
    

    
    # 正常训练
    def train(self):
        # print(self.id,end=' ')
        # sys.stdout.flush()
        self.model.to(self.args.device)
        self.model.train()
        self.train_data.x = self.train_data.x.to(self.args.device)
        self.train_data.edge_index = self.train_data.edge_index.to(self.args.device)

        max_local_epochs = self.local_epochs
        self.train_edge_label = self.train_edge_label.to(self.args.device)

        rdp = 0
        orders = (list(range(2, 64)) + [128, 256, 512])  # 默认的lamda
        epsilon_list = []
        iterations = 1
        delta = 10 ** (-5)

        for step in range(max_local_epochs):
            # if self.args.defense:
            #     self.optimizer.zero_grad()
            #     output = self.model(self.train_data.x,self.train_data.edge_index,self.train_edge_label_index)
            #     # output = output.to(self.args.device)
            #     all_index = [i for i in range(self.train_edge_label.size()[0])]
            #     batch_index = random.choices(all_index, k=self.batch_size)

            #     for i in batch_index:
            #         loss = self.loss(output[i],self.train_edge_label[i])
            #         loss.backward(retain_graph=True)
            #     self.optimizer.step_dp()

            #     rdp_every_epoch=compute_rdp(self.batch_size/self.num_nodes, self.sigma, 1*iterations, orders)
            #     rdp=rdp+rdp_every_epoch
            #     epsilon, best_alpha = compute_eps(orders, rdp, delta)
            #     epsilon_list.append(epsilon)
            #     print("epoch: {:3.0f}".format(step + 1) + " | epsilon: {:7.4f}".format(
            #         epsilon) + " | best_alpha: {:7.4f}".format(best_alpha)  )
            #     # pred = output.argmax(dim=1)  # Use the class with highest probability.
            #     # test_correct = pred[self.dataset.test_mask] == self.dataset.y[self.dataset.test_mask]  # Check against ground-truth labels.
            #     # test_acc = int(test_correct.sum()) / int(self.dataset.test_mask.sum()) 
            # else:
            self.optimizer.zero_grad()
            output = self.model(self.train_data.x,self.train_data.edge_index,self.train_edge_label_index)
            # output = output.to(self.args.device)
            loss = self.loss(output,self.train_edge_label)
            loss.backward()
            self.optimizer.step()

            # if self.args.defense:
            #     rdp_every_epoch=compute_rdp(1, self.sigma, 1*iterations, orders)
            #     rdp=rdp+rdp_every_epoch
            #     epsilon, best_alpha = compute_eps(orders, rdp, delta)
            #     epsilon_list.append(epsilon)
            #     print("epoch: {:3.0f}".format(step + 1) + " | epsilon: {:7.4f}".format(
            #         epsilon) + " | best_alpha: {:7.4f}".format(best_alpha)  )
                

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()   

    

    def get_abs_mean(self,grads):
        total_abs_sum = 0.0
        total_grads = 0
        layer = [0,1,2,3,4,5]
        for i ,grad in enumerate(grads):
            if i in layer:
                total_abs_sum+=grad.abs().sum()
                total_grads+=grad.numel()

        average_abs_sum = total_abs_sum / total_grads
        
        return total_abs_sum
    
    def train_gradient(self):
        print(self.id)
        # sys.stdout.flush()
        self.model.to(self.args.device)

        self.pre_model = copy.deepcopy(self.model)
        self.batch_gradient = copy.deepcopy(self.model)
        for params in self.batch_gradient.parameters():
            params.data.zero_()

        # 计算分布情况
        self.cal_dis()
        self.model.train()
        max_local_epochs = self.local_epochs
        self.train_data.x = self.train_data.x.to(self.args.device)
        self.train_data.edge_index = self.train_data.edge_index.to(self.args.device)

        # if self.args.defense:
        #     self.label_distribution = [0 for i in range(2)]

        rdp = 0
        orders = (list(range(2, 64)) + [128, 256, 512])  # 默认的lamda
        epsilon_list = []
        iterations = 1
        delta = 10 ** (-5)

        for step in range(max_local_epochs):
            # if self.args.defense:
                
            #     self.optimizer.zero_microbatch_grad()
            #     output = self.model(self.train_data.x,self.train_data.edge_index,self.train_edge_label_index)
            #     # output = output.to(self.args.device)
            #     a = self.train_edge_label
            #     all_index = [i for i in range(self.train_edge_label.size()[0])]
            #     batch_index = random.choices(all_index, k=self.batch_size)
            #     self.cal_dis_batch(batch_index)
            #     for i in batch_index:
            #         loss = self.loss(output[i],self.train_edge_label[i])
            #         loss.backward(retain_graph=True)
            #         self.optimizer.microbatch_step()
            #     self.optimizer.step_dp()
            #     tmp_grad = []
            #     for batch_gradient_params,params in zip(self.batch_gradient.parameters(),self.model.parameters()):
            #         # params.grad.data.clamp_()
            #         tmp_grad.append(params.grad.data.clone().detach())
            #         batch_gradient_params.data+=params.grad.data.clone().detach()
                
            #     rdp_every_epoch=compute_rdp(self.batch_size/self.num_nodes, self.sigma, 1*iterations, orders)
            #     rdp=rdp+rdp_every_epoch
            #     epsilon, best_alpha = compute_eps(orders, rdp, delta)
            #     epsilon_list.append(epsilon)
            #     print("epoch: {:3.0f}".format(step + 1) + " | epsilon: {:7.4f}".format(
            #         epsilon) + " | best_alpha: {:7.4f}".format(best_alpha)  )
                
            # else:    
            self.optimizer.zero_grad()
            # for params in self.model.parameters():
            #     print(params.grad)
            output = self.model(self.train_data.x,self.train_data.edge_index,self.train_edge_label_index)
            loss = self.loss(output,self.train_edge_label)
            loss.backward()
            # if self.args.model_gradient_clip:
            #     self.clip_gradient()
            # tmp_grad = []
            for batch_gradient_params,params in zip(self.batch_gradient.parameters(),self.model.parameters()):
                # params.grad.data.clamp_()
                # tmp_grad.append(params.grad.data.clone().detach())
                batch_gradient_params.data+=params.grad.data.clone().detach()
            # print(' epoch ',step,'  avg abs grad: ',self.get_abs_mean(tmp_grad))
            self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

    def cal_dis(self):
        if self.args.defense:
       
            self.train_edge_label_index
            self.train_edge_label
            hit_num = 0
            for i in range(len(self.train_edge_label_index[0])):
                src = self.train_edge_label_index[0][i].item()
                end = self.train_edge_label_index[1][i].item()
                for j in range(len(self.ori_train_data_dege_index[0])):
                    if (self.ori_train_data_dege_index[0][j] == src and self.ori_train_data_dege_index[1][j] == end) \
                        or (self.ori_train_data_dege_index[1][j] == src and self.ori_train_data_dege_index[0][j] == end):
                        hit_num+=1
                        break
            miss_num = len(self.train_edge_label) - hit_num
            self.label_distribution = [miss_num/len(self.train_edge_label_index[0]),hit_num/len(self.train_edge_label_index[0])]
        else:
            self.label_distribution = [self.train_negative_links/self.train_links_num,self.train_positive_links/self.train_links_num]

    
    
    def cal_dis_batch(self,batch_index):

        for i in batch_index:
            self.label_distribution[self.train_edge_label[i]]+=1
        self.label_distribution=[i/len(batch_index) for i in self.label_distribution]

    def get_gradient(self):
        ret = []
        for params in self.batch_gradient.parameters():
            # 尝试分母乘以跳数
            # ret.append(params.data.clone().detach()*(1/self.local_epochs))
            if self.local_epochs == 1:
                ret.append(params.data.clone().detach())
            else:
                # ret.append(params.data.clone().detach()*(1/(self.local_epochs*(self.args.gcn_hops+3))))
                ret.append(params.data.clone().detach()*(1/self.args.local_epochs))
            # ret.append(params.data.clone().detach()*(1/10))
        return ret

    def get_label_distribution(self):
        return np.array(self.label_distribution)