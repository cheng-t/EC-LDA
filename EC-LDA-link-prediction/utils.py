import torch
from torch_geometric.utils import to_dense_adj ,dense_to_sparse


def perturb_adj_laplace(adj_edge_index,eps):   #adj本身没有自连接的
    adj = to_dense_adj(adj_edge_index)[0]
    tmp_adj = adj
    # adj本身没有自连接的，对角线本身为0，而且我们传回去的邻接矩阵不需要构建自连接，后面函数会构建
    N = adj.shape[0]
    for i in range(N):
        adj[i][i]=0

    # count_1 = 0
    # for i in range(N):
    #     for j in range(N):
    #         if adj[i][j]==1:
    #             count_1+=1

    eps_e=eps*0.1
    eps_l=eps-eps_e

    edge_num = torch.count_nonzero(adj).item()
    lap_noise = torch.distributions.Laplace(0, 1.0 / eps_e).sample((1,))

    top_k=int(edge_num+lap_noise)
    while top_k <= 0:
        lap_noise = torch.distributions.Laplace(0, 1.0 / eps_e).sample((1,))
        top_k=int(edge_num+lap_noise)
    
    perturbed_matrix = torch.zeros(N, N)

    laplace_noise = torch.distributions.Laplace(0, 1 / eps_l).sample(adj.size())
    adj_perturbed = adj + laplace_noise
    # 将adj_perturbed展开为一维张量，以便统一处理
    adj_perturbed_flat = adj_perturbed.flatten()

    # 移除对角线元素，防止其被选为top k元素
    mask = ~torch.eye(N, dtype=bool).flatten()
    adj_perturbed_flat_no_diag = adj_perturbed_flat[mask]

    # 使用topk找到非对角线前non_diag_edge_num个元素的值及其索引
    topk_values, topk_indices = torch.topk(adj_perturbed_flat_no_diag, k=top_k, largest=True)

    # 将一维索引转换回二维索引
    flat_indices_no_diag = torch.arange(N * N)[mask]
    actual_topk_indices = flat_indices_no_diag[topk_indices]
    row_indices = actual_topk_indices // N
    col_indices = actual_topk_indices % N

    # 使用高级索引将perturbed_matrix中对应位置置1
    perturbed_matrix[row_indices, col_indices] = 1.0

    # for i in range(N):
    #     perturbed_matrix[i][i]=1
    
    # count_1 = 0
    # for i in range(N):
    #     for j in range(N):
    #         if perturbed_matrix[i][j]==1:
    #             count_1+=1

    # count = 0
    # for i in range(N):
    #     for j in range(N):
    #         if tmp_adj[i][j]==1 and perturbed_matrix[i][j] == 1:
    #             count+=1
    # all = N**2
    # print(count)
    ret, __ = dense_to_sparse(perturbed_matrix)
    return ret


class Edge:
    def __init__(self,src,end):
        self.src = src
        self.end = end