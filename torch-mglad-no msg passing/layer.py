import torch
import torch.nn as nn
import torch.nn.functional as F


class MGLADlayer(nn.Module):
    def __init__(self, num_nodes, wkr_dim, tsk_dim, num_rels, num_wkr, num_tsk, e_dim):
        super(MGLADlayer, self).__init__()
        self.num_nodes = num_nodes
        self.wkr_dim = wkr_dim
        self.tsk_dim = tsk_dim
        self.num_rels = num_rels
        self.num_wkr = num_wkr
        self.num_tsk = num_tsk
        self.e_dim = e_dim

        # self.weight_worker = nn.Parameter(torch.Tensor(num_rels, tsk_dim, wkr_dim))
        # self.weight_task = nn.Parameter(torch.Tensor(num_rels, wkr_dim, tsk_dim))

        self.mlp_tsk = MLP(tsk_dim, tsk_dim)
        #self.mlp_tsk_inner = MLP(wkr_dim + e_dim, e_dim)

        self.mlp_wkr = MLP(wkr_dim, wkr_dim)
        #self.mlp_wkr_inner = MLP(tsk_dim + e_dim, e_dim)

        self.e = nn.Parameter(torch.Tensor(num_rels, 1, e_dim))  # 1*k维
        nn.init.xavier_uniform_(self.e, gain=nn.init.calculate_gain('relu'))

        # nn.init.xavier_uniform_(self.weight_worker, gain=nn.init.calculate_gain('relu'))
        # nn.init.xavier_uniform_(self.weight_task, gain=nn.init.calculate_gain('relu'))

    def forward(self, g):
        ability_new=self.mlp_wkr(g.ndata['ability'])
        g.ndata['ability']=ability_new
        labels_new = self.mlp_tsk(g.ndata['labels'])
        g.ndata['labels']=labels_new
        # # squeeze()最好都加上dim
        # def msg_func_wkr(edges):
        #     tau_i = edges.src['labels'].squeeze(dim=1)  # 4212,2
        #     e_l = self.e[edges.data['type'].long()].squeeze(dim=1)  # 4212,e_dim
        #     msg = self.mlp_wkr_inner(torch.cat((tau_i, e_l), dim=1).unsqueeze(1))  # 4212,1,e_dim
        #     return {'msg_ability': msg}
        #
        # def red_func_wkr(nodes):
        #     M_i = torch.div(torch.sum(nodes.mailbox['msg_ability'], dim=1), nodes.data['deg']).squeeze(dim=1)  # 求mean
        #     a_j = nodes.data['ability'].squeeze(dim=1)
        #     return {'ability': self.mlp_wkr(torch.cat((a_j, M_i), dim=1).unsqueeze(1))}
        #
        # def msg_func_tsk(edges):
        #     a_j = edges.src['ability'].squeeze(dim=1)
        #     e_l = self.e[edges.data['type'].long()].squeeze(dim=1)
        #     msg = self.mlp_tsk_inner(torch.cat((a_j, e_l), dim=1).unsqueeze(1))
        #     return {'msg_tau': msg}
        #
        # def red_func_tsk(nodes):
        #     M_i = torch.div(torch.sum(nodes.mailbox['msg_tau'], dim=1), nodes.data['deg']).squeeze(dim=1)  # 求mean
        #     tau_i = nodes.data['labels'].squeeze(dim=1)
        #     return {'labels': self.mlp_tsk(torch.cat((tau_i, M_i), dim=1).unsqueeze(1))}
        #
        # # 更新 worker 的特征
        # g.register_message_func(msg_func_wkr)
        # g.register_reduce_func(red_func_wkr)
        # g.pull(g.nodes()[range(self.num_wkr)])  # pull 到 wkr 节点
        #
        # # 更新 task 的特征
        # g.register_message_func(msg_func_tsk)
        # g.register_reduce_func(red_func_tsk)
        # g.pull(g.nodes()[range(self.num_wkr, self.num_nodes)])  # pull 到 tsk 节点
        #
        # # for param in self.mlp_wkr.named_parameters():
        # #     print(param)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super(MLP, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias)
        # 一层fc 有bias

    def forward(self, d_input):
        return F.relu(self.fc(d_input))
