import torch
import torch.nn as nn
import torch.nn.functional as F


class MGLADlayer(nn.Module):
    def __init__(self, num_nodes, wkr_dim, tsk_dim, num_rels, num_wkr, num_tsk):
        super(MGLADlayer, self).__init__()
        self.num_nodes = num_nodes
        self.wkr_dim = wkr_dim
        self.tsk_dim = tsk_dim
        self.num_rels = num_rels
        self.num_wkr = num_wkr
        self.num_tsk = num_tsk

        self.mlp_tsk = MLP(tsk_dim + wkr_dim, tsk_dim)

        self.mlp_wkr = MLP(wkr_dim + tsk_dim, wkr_dim)

        self.edge_weight = nn.Parameter(torch.Tensor(num_rels, 1, 1))
        nn.init.xavier_uniform_(self.edge_weight, gain=nn.init.calculate_gain('relu'))

        # self.weight_worker = nn.Parameter(torch.Tensor(num_rels, tsk_dim, wkr_dim))
        # self.weight_task = nn.Parameter(torch.Tensor(num_rels, wkr_dim, tsk_dim))

        # nn.init.xavier_uniform_(self.weight_worker, gain=nn.init.calculate_gain('relu'))
        # nn.init.xavier_uniform_(self.weight_task, gain=nn.init.calculate_gain('relu'))

    def forward(self, g):
        # squeeze()最好都加上dim
        def msg_func_wkr(edges):
            tau_i = edges.src['labels']  # 4212,1,2
            weight_l = self.edge_weight[edges.data['type'].long()]  # 4212,1,1
            msg = tau_i.mul(weight_l)  # 4212,1,2
            return {'msg_ability': msg}

        def red_func_wkr(nodes):
            message = torch.div(torch.sum(nodes.mailbox['msg_ability'], dim=1), nodes.data['deg']).squeeze(
                dim=1)  # 求mean   # 39,2
            # message = torch.sum(nodes.mailbox['msg_ability'], dim=1).squeeze(
            #     dim=1)  # 求mean   # 39,2
            a_j = nodes.data['ability'].squeeze(dim=1)
            return {'ability': self.mlp_wkr(torch.cat((a_j, message), dim=1)).unsqueeze(1)}

        def msg_func_tsk(edges):
            a_j = edges.src['ability']
            weight_l = self.edge_weight[edges.data['type'].long()]
            msg = a_j.mul(weight_l)
            return {'msg_tau': msg}

        def red_func_tsk(nodes):
            message = torch.div(torch.sum(nodes.mailbox['msg_tau'], dim=1), nodes.data['deg']).squeeze(dim=1)  # 求mean
            # message = torch.sum(nodes.mailbox['msg_tau'], dim=1).squeeze(dim=1)  # 求mean
            tau_i = nodes.data['labels'].squeeze(dim=1)
            return {'labels': self.mlp_tsk(torch.cat((tau_i, message), dim=1)).unsqueeze(1)}

        # 更新 worker 的特征
        g.register_message_func(msg_func_wkr)
        g.register_reduce_func(red_func_wkr)
        g.pull(g.nodes()[range(self.num_wkr)])  # pull 到 wkr 节点

        # 更新 task 的特征
        g.register_message_func(msg_func_tsk)
        g.register_reduce_func(red_func_tsk)
        g.pull(g.nodes()[range(self.num_wkr, self.num_nodes)])  # pull 到 tsk 节点

        # for param in self.mlp_wkr.named_parameters():
        #     print(param)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super(MLP, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias)
        # 一层fc 有bias

    def forward(self, d_input):
        return F.relu(self.fc(d_input))
