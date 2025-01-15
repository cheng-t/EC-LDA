
from attacks.batch_label_reconstruction import batch_label_construction,batch_label_construction_output_layer
from iLRG.iLRG import batch_label_construction_iLRG
from LLG_star_random.LLG_star import batch_label_construction_LLG_star
from FLcore.client.clientfedavg import clientAVG
from FLcore.client.clientfedavgblra import clientAVGblra
from FLcore.server.serverbase import Server
import sys
import numpy as np
import torch
import copy
from result.view_attack import attack_performance,mean_params_and_metric_and_random,acc_with_global_round
import matplotlib.pyplot as plt

def compare_models(model1,model2):
    for params1,params2 in zip(model1.parameters(),model2.parameters()):
        if not torch.equal(params1,params2):
            print('参数不相同！')
            return False
    print('两个模型参数完全相同')
    return True


def mean_params_and_l2_norm(a,b,random):
    # 创建画布和第一个子图  
    fig, ax1 = plt.subplots()  
    
    # 绘制第一个数据集  
    ax1.plot(a, 'g-')  
    ax1.set_xlabel('X ')  
    ax1.set_ylabel('Y1 mean_params', color='g')  
    
    # 创建第二个子图并绑定到同一X轴  
    ax2 = ax1.twinx()  
    ax2.plot(b, 'b-')  
    ax2.set_ylabel('Y2 metric', color='b')  
    ax2.plot(random)
    
    plt.savefig("save/image.png")
    # 显示图形  
    plt.show()  


def mean_of_params(model):
    # 计算所有参数的绝对值之和的平均
    total_abs_sum = 0.0
    total_params = 0

    for i,param in enumerate(model.parameters()):
        # if i == 1:
        if True:
            total_abs_sum += param.abs().sum()
            total_params += param.numel()

    average_abs_sum = total_abs_sum / total_params
    return average_abs_sum



def mean_of_grad(grads):
    total_abs_sum = 0.0
    total_grads = 0
    layer = 2
    for i ,grad in enumerate(grads):
        # if i == layer:
        if True:
            total_abs_sum+=grad.abs().sum()
            total_grads+=grad.numel()

    average_abs_sum = total_abs_sum / total_grads
    return average_abs_sum

def metrics(restored_dis,true_dis,args,metric=None):
    res = None
    flag=True
    for i in restored_dis:
        if i!=0:
            flag=False
    if flag:
        restored_dis = [1e-64 for i in range(len(restored_dis))]
    return_metric = args.metric if metric == None else metric
    if return_metric == 'l2_norm':
        res = np.linalg.norm(restored_dis-true_dis)
    elif return_metric == 'cosine_similarity':
        res = np.dot(restored_dis,true_dis)/(np.linalg.norm(restored_dis)*np.linalg.norm(true_dis))
    elif return_metric == 'js_div':
        res_dis = np.array([a if a!=0.0 else 1e-10 for a in restored_dis])
        tre_dis = np.array([a if a!=0.0 else 1e-10 for a in true_dis])
        M = (res_dis+tre_dis)/2
        res = 0.5 * np.sum(res_dis*np.log(res_dis/M)) + 0.5 * np.sum(tre_dis*np.log(tre_dis/M))
    elif return_metric == 'hellinger_dis':
        # M = (res_dis+tre_dis)/2
        res = 1/np.sqrt(2)*np.linalg.norm(np.sqrt(restored_dis)-np.sqrt(true_dis))
    else:
        print('No such Metric!')
        exit()

    return res


def attack(client,cur_metric,cur_metric_random,args,num_classes):
    gradient = client.get_gradient()
    avg_grad = mean_of_grad(copy.deepcopy(gradient))
    client.batch_gradient=None
    train_data = client.dataset
    # restored_dis,O = batch_label_construction_output_layer(args,client.pre_model,gradient,client.num_nodes,feature_size=16)
    if args.compare:
        restored_dis_no_active,O, sub,var  = batch_label_construction(None,args,client.pre_model,gradient,client.num_nodes,feature_size = args.feature_size,dis_y = None)
        restored_dis_iLRG = batch_label_construction_iLRG(args,client.pre_model,gradient,client.num_nodes)
        restored_dis_LLG_star = batch_label_construction_LLG_star(args,client.pre_model,gradient,client.num_nodes)
        O = 0
        sub = 0
        var = 0
    else:
        restored_dis,O, sub,var  = batch_label_construction(None,args,client.pre_model,gradient,client.num_nodes,feature_size = args.feature_size,dis_y = None)
    # restored_dis,O = batch_label_construction(None,args,client.pre_model,gradient,client.num_nodes,feature_size = 1433,dis_y = restored_dis)
    true_dis = client.get_label_distribution()

    if args.compare:
        metric_cos_sim_no_active = metrics(restored_dis_no_active,true_dis,args,"cosine_similarity")
        metric_cos_sim_iLRG = metrics(restored_dis_iLRG,true_dis,args,"cosine_similarity")
        metric_cos_sim_LLG_star = metrics(restored_dis_LLG_star,true_dis,args,"cosine_similarity")

        metric_js_div_no_active = metrics(restored_dis_no_active,true_dis,args,"js_div")
        metric_js_div_iLRG = metrics(restored_dis_iLRG,true_dis,args,"js_div")
        metric_js_div_LLG_star = metrics(restored_dis_LLG_star,true_dis,args,"js_div")
        cur_metric.append([[metric_cos_sim_no_active,metric_cos_sim_iLRG,metric_cos_sim_LLG_star],
                          [metric_js_div_no_active,metric_js_div_iLRG,metric_js_div_LLG_star]])
        metric_cos_sim = -1
        metric_js_div = -1
    else:
        metric_cos_sim = metrics(restored_dis,true_dis,args,"cosine_similarity")
        metric_js_div = metrics(restored_dis,true_dis,args,"js_div")
        print('\n恢复分布：',restored_dis)
        print('真实分布：',true_dis)
        print('metric {}:{:.4f}'.format('cosine_similarity',metric_cos_sim),end = ' ')
        print('metric {}:{:.4f}'.format('js_div',metric_js_div))
        cur_metric.append([metric_cos_sim,metric_js_div])

    random_guess = np.array([1/num_classes for i in range(num_classes)])
    # l2_norm_random = np.linalg.norm(random_guess-true_dis)

    metric_random_cos_sim = metrics(random_guess,true_dis,args,'cosine_similarity')
    metric_random_js_div = metrics(random_guess,true_dis,args,'js_div')

    cur_metric_random.append([metric_random_cos_sim,metric_random_js_div]) 
    return O,avg_grad,[metric_cos_sim,metric_js_div],sub,var

def report_attack(cur_metric,cur_metric_random,all_metric,all_random,args):
    # cur_metric_mean = sum(cur_metric)/len(cur_metric)
    if not args.compare:
        cur_metric_cos_sim = [row[0] for row in cur_metric]
        cur_metric_js_div = [row[1] for row in cur_metric]
        cur_metric_mean_cos_sim = sum(cur_metric_cos_sim)/len(cur_metric_cos_sim)
        cur_metric_mean_js_div = sum(cur_metric_js_div)/len(cur_metric_js_div)
        # cur_metric_mean_random = sum(cur_metric_random)/len(cur_metric_random)
        cur_metric_random_cos_sim = [row[0] for row in cur_metric_random]
        cur_metric_random_js_div = [row[1] for row in cur_metric_random]
        cur_metric_mean_random_cos_sim = sum(cur_metric_random_cos_sim)/len(cur_metric_random_cos_sim)
        cur_metric_mean_random_js_div = sum(cur_metric_random_js_div)/len(cur_metric_random_js_div)
        all_metric.append([cur_metric_mean_cos_sim,cur_metric_mean_js_div])
        all_random.append([cur_metric_mean_random_cos_sim,cur_metric_mean_random_js_div])
        print('\nmetric: cos-sim',"本轮的攻击与真实分布的平均metric为:",cur_metric_mean_cos_sim)
        print('metric: js_div',"本轮的攻击与真实分布的平均metric为:",cur_metric_mean_js_div)
        
        print('metric: cos-sim',"本轮随机猜测与真实分布的平均metric为:",cur_metric_mean_random_cos_sim)
        print('metric: js_div',"本轮随机猜测与真实分布的平均metric为:",cur_metric_mean_random_js_div)
    else:
        cur_metric_cos_sim = [row[0] for row in cur_metric]
        cur_metric_js_div = [row[1] for row in cur_metric]
        cur_metric_cos_sim_no_active = [row[0] for row in cur_metric_cos_sim]
        cur_metric_cos_sim_iLRG = [row[1] for row in cur_metric_cos_sim]
        cur_metric_cos_sim_LLG_star = [row[2] for row in cur_metric_cos_sim]
        cur_metric_js_div_no_active = [row[0] for row in cur_metric_js_div]
        cur_metric_js_div_iLRG = [row[1] for row in cur_metric_js_div]
        cur_metric_js_div_LLG_star = [row[2] for row in cur_metric_js_div]

        cur_metric_mean_cos_sim_no_active = sum(cur_metric_cos_sim_no_active)/len(cur_metric_cos_sim_no_active)
        cur_metric_mean_cos_sim_iLRG = sum(cur_metric_cos_sim_iLRG)/len(cur_metric_cos_sim_iLRG)
        cur_metric_mean_cos_sim_LLG_star = sum(cur_metric_cos_sim_LLG_star)/len(cur_metric_cos_sim_LLG_star)

        cur_metric_mean_js_div_no_active = sum(cur_metric_js_div_no_active)/len(cur_metric_js_div_no_active)
        cur_metric_mean_js_div_iLRG = sum(cur_metric_js_div_iLRG)/len(cur_metric_js_div_iLRG)
        cur_metric_mean_js_div_LLG_star = sum(cur_metric_js_div_LLG_star)/len(cur_metric_js_div_LLG_star)

        all_metric.append([[cur_metric_mean_cos_sim_no_active,cur_metric_mean_cos_sim_iLRG,cur_metric_mean_cos_sim_LLG_star],
                          [cur_metric_mean_js_div_no_active,cur_metric_mean_js_div_iLRG,cur_metric_mean_js_div_LLG_star]])
        
        print('\nmetric: cos-sim',"LDA 本轮的攻击与真实分布的平均metric为:",cur_metric_mean_cos_sim_no_active)
        print('metric: cos-sim',"iLRG 本轮的攻击与真实分布的平均metric为:",cur_metric_mean_cos_sim_iLRG)
        print('metric: cos-sim',"LLG_star 本轮的攻击与真实分布的平均metric为:",cur_metric_mean_cos_sim_LLG_star)

        print('metric: js_div',"LDA 本轮的攻击与真实分布的平均metric为:",cur_metric_mean_js_div_no_active)
        print('metric: js_div',"iLRG 本轮的攻击与真实分布的平均metric为:",cur_metric_mean_js_div_iLRG)
        print('metric: js_div',"LLG_star 本轮的攻击与真实分布的平均metric为:",cur_metric_mean_js_div_LLG_star)


def report_all_attack(attack_round,metric,args):
    print('metric:',args.metric,' 攻击结果为：')
    if len(attack_round)==0:
        print('本次实验未进行攻击！')

    if not args.compare:
        for i in range(len(attack_round)):
            print('攻击轮次：{:3}   metric:cos-sim: {:.5f}, js-div: {:.5f}'.format(attack_round[i],metric[i][0],metric[i][1]))
    else:
        print("LDA")
        for i in range(len(attack_round)):
            print('攻击轮次：{:3}   metric:cos-sim: {:.5f}, js-div: {:.5f}'.format(attack_round[i],metric[i][0][0],metric[i][1][0]))
            # print('攻击轮次：{:3}   本轮的攻击与真实分布的平均metric为: {}'.format(attack_round[i],metric[i]))
        print('iLRG')
        for i in range(len(attack_round)):
            print('攻击轮次：{:3}   metric:cos-sim: {:.5f}, js-div: {:.5f}'.format(attack_round[i],metric[i][0][1],metric[i][1][1]))

        print('LLG_star')
        for i in range(len(attack_round)):
            print('攻击轮次：{:3}   metric:cos-sim: {:.5f}, js-div: {:.5f}'.format(attack_round[i],metric[i][0][2],metric[i][1][2]))


class FedAvgBLRA(Server):
    def __init__(self, args):
        super().__init__(args)
        self.set_clients(clientAVGblra)
        # self.clip_client()
        # self.victim = [i for i in range(self.args.num_clients)]
        self.victim = args.victim_client
        self.pre_model = copy.deepcopy(self.global_model)
    

    # clip全局模型的参数
    # TODO:裁剪之后再加上一个较小的正态分布噪声，观察攻击效果
    def active_attack_clip_params(self):
        # for params in self.global_model.parameters():
        #     # 0.0001的效果很好
        #     # params.data.clamp_(-0.05,0.05) 
        #     params.data.clamp_(-0.0001,0.0001)
        #     # params.data*=10
        #     # print(params.data)  
        #     # noise = torch.normal(mean=0.0,std=0.1,size = params.data.size())
        #     # params.data.add_(noise)
        #     # print(params.data)

        params_dict = self.global_model.state_dict()
        dict_size = len(params_dict)
        # for i,params in enumerate(self.global_model.parameters()):
        #     if i==dict_size-1:
        #         params.data.clamp_(-0.0001,0.0001)
        #         noise = torch.normal(mean=0.0,std=0.001,size = params.data.size())
        #         params.data.add_(noise)

        # todo:范数裁剪
        total_norm = 0
        for i,params in enumerate(self.global_model.parameters()):
            if params.requires_grad:
                total_norm+=params.data.norm(2).item() ** 2.
        total_norm = total_norm ** .5
        # clip_coef = min(self.l2_norm_clip / (total_norm + 1e-6), 1.)
        clip_coef = min(self.args.clip_norm / (total_norm + 1e-6), 1.)
        
        param_clip_list = [dict_size-4,dict_size-3]
        for i,params in enumerate(self.global_model.parameters()):
            # if i in param_clip_list:
            if True:
                if params.requires_grad:
                    params.data.mul_(clip_coef)
                    # params.data.mul_(1e-5)
        return clip_coef

    def set_num_classes(self,args):
        args.num_classes=2
        for client in self.clients:
            client.num_classes=2
        self.args.num_classes=2

    def train(self):
        all_metric = []
        mean_params = []
        all_random = []
        all_clip_coef = []
        all_O = []
        all_O_metric = []
        self.set_num_classes(self.args)
        all_acc = []
        for i in range(self.global_rounds):
            sys.stdout.flush()

            self.selected_clients = self.select_clients()

            # 保存上一轮的全局模型
            self.pre_model = copy.deepcopy(self.global_model)
            
            if not self.args.compare and i in self.args.attack_global_round:
                clip_coef = self.active_attack_clip_params()
                all_clip_coef.append(clip_coef)
                # a=0
                
            self.send_models()
            cur_metric = []
            cur_metric_random = []
            cur_O = []
            
            print('\nExperiment ',self.args.cur_experiment_no,end='')
            print(f"-------------Round number: {i}-------------")

            for client in self.selected_clients:
                # if client.id in self.victim :

                if client.id in self.victim and i in self.args.attack_global_round:
                    client.train_gradient()
                    O,avg_grad,O_metric,sub,var=attack(client,cur_metric,cur_metric_random,self.args,client.num_classes)
                    O_metric = [O,O_metric,sub,var]
                    all_O_metric.append(O_metric) 
                    # print(" O的值为:",O)
                    # print('avg grad:',avg_grad.item())
                    mean_param = mean_of_params(self.global_model)
                    # mean_params.append(avg_grad.item())
                    mean_params.append(mean_param.item())
                    cur_O.append(O)
                else:
                    client.train()
                    
            if self.args.model_params_clip:
                self.clip_client()

            # if i % self.eval_gap == 0 :  # 几轮测试一次全局模型
            # # if i % self.eval_gap == 0 and i in self.args.attack_global_round:  # 几轮测试一次全局模型
            #     print(f"\n-------------Round number: {i}-------------")
            #     # print("\nEvaluate global model")
            #     self.evaluate()
            #     if i in self.args.attack_global_round:
            #         report_attack(cur_l2_norm,cur_l2_norm_random,all_l2_norm)
            #     # mean = mean_of_params(self.global_model)
            #     # print('本轮模型参数的绝对值平均为：',mean.item())
            #     # mean_params.append(mean.item())

            self.receive_models()
            self.aggregate_parameters()
            # print(cur_metric)
            # if i in self.args.attack_global_round:
            #     print('\n主动攻击的分类结果：',i)
            #     self.evaluate()
            if not self.args.compare and i in self.args.attack_global_round:
                self.global_model = copy.deepcopy(self.pre_model)
                
            
            if i % self.eval_gap == 0 :  # 几轮测试一次全局模型
            # if i % self.eval_gap == 0 and i in self.args.attack_global_round:  # 几轮测试一次全局模型
                # print(f"\n-------------Round number: {i}-------------")
                self.send_models()
                print("\nEvaluate global model")
                all_acc.append(self.evaluate())
                if i in self.args.attack_global_round:
                    report_attack(cur_metric,cur_metric_random,all_metric,all_random, self.args)
                # mean = mean_of_params(self.global_model)
                # print('本轮模型参数的绝对值平均为：',mean.item())
                # mean_params.append(mean.item())
            all_O.append(cur_O)
        
        
        self.send_models()
        print('')
        print('Finish')
        final_acc= self.evaluate()
        
        print('Best Acc:','{: .6f}'.format(self.best_acc))
        print('Final Acc:','{: .6f}'.format(final_acc))
        print('victim client:',self.args.victim_client)
        # mean_params_and_metric_and_random(mean_params,all_metric,all_random)
        acc_with_global_round(all_acc)
        # mean_params_and_l2_norm(mean_params,all_metric,all_random)
        
        # attack_performance(self.args.attack_global_round,all_metric,all_random)
        report_all_attack(self.args.attack_global_round,all_metric,self.args)
        if not self.args.compare:
            print(all_metric)
            if self.args.compare:
                print('method: ',self.args.compare_name)
            else:
                print('method: ALDIA')
            all_metric_cos_sim = [row[0] for row in all_metric]
            all_metric_js_div = [row[1] for row in all_metric]
            avg_metric_cos_sim = sum(all_metric_cos_sim)/len(all_metric_cos_sim)
            avg_metric_js_div =  sum(all_metric_js_div)/len(all_metric_js_div)

            print('mean of metric cos-sim:','{: .6f}'.format(sum(all_metric_cos_sim)/len(all_metric_cos_sim)))
            print('mean of metric cos-sim','{}'.format(sum(all_metric_cos_sim)/len(all_metric_cos_sim)))

            print('mean of metric js-div:','{: .6f}'.format(sum(all_metric_js_div)/len(all_metric_js_div)))
            print('mean of metric js-div','{}'.format(sum(all_metric_js_div)/len(all_metric_js_div)))
            # print('\nO:',all_O)
            # print(all_clip_coef)
            # for o_m in all_O_metric:
            #     print(o_m)

            sum_sub = 0
            num = 0
            for i in all_O_metric:
                sum_sub+=i[2]
                num+=1
            mean_sub = sum_sub/num
            print('mean_sub:{:.6f}'.format(mean_sub))

            sum_var = 0
            num=0
            for i in all_O_metric:
                sum_var+=i[3]*1e10
                num+=1
            mean_var = sum_var/num
            print('mean_var:{:.6f}'.format(mean_var))

            return avg_metric_cos_sim, avg_metric_js_div, final_acc, all_acc
        else:
            all_metric_cos_sim_no_active = [row[0][0] for row in all_metric]
            all_metric_cos_sim_iLRG = [row[0][1] for row in all_metric]
            all_metric_cos_sim_LLG_star = [row[0][2] for row in all_metric]

            all_metric_js_div_no_active = [row[1][0] for row in all_metric]
            all_metric_js_div_iLRG = [row[1][1] for row in all_metric]
            all_metric_js_div_LLG_star = [row[1][2] for row in all_metric]

            avg_metric_cos_sim_no_active = sum(all_metric_cos_sim_no_active)/len(all_metric_cos_sim_no_active)
            avg_metric_cos_sim_iLRG = sum(all_metric_cos_sim_iLRG)/len(all_metric_cos_sim_iLRG)
            avg_metric_cos_sim_LLG_star = sum(all_metric_cos_sim_LLG_star)/len(all_metric_cos_sim_LLG_star)

            avg_metric_js_div_no_active =  sum(all_metric_js_div_no_active)/len(all_metric_js_div_no_active)
            avg_metric_js_div_iLRG =  sum(all_metric_js_div_iLRG)/len(all_metric_js_div_iLRG)
            avg_metric_js_div_LLG_star =  sum(all_metric_js_div_LLG_star)/len(all_metric_js_div_LLG_star)
            
            print('cos-sim')
            print('LDA mean of metric cos-sim:','{: .6f}'.format(avg_metric_cos_sim_no_active))
            print('iLRG mean of metric cos-sim:','{: .6f}'.format(avg_metric_cos_sim_iLRG))
            print('LLG_star mean of metric cos-sim:','{: .6f}'.format(avg_metric_cos_sim_LLG_star))

            print('js_div')
            print('LDA mean of metric js_div:','{: .6f}'.format(avg_metric_js_div_no_active))
            print('iLRG mean of metric js_div:','{: .6f}'.format(avg_metric_js_div_iLRG))
            print('LLG_star mean of metric js_div:','{: .6f}'.format(avg_metric_js_div_LLG_star))
            
            return [avg_metric_cos_sim_no_active,avg_metric_cos_sim_iLRG,avg_metric_cos_sim_LLG_star], \
                    [avg_metric_js_div_no_active,avg_metric_js_div_iLRG,avg_metric_js_div_LLG_star], \
                    final_acc, all_acc