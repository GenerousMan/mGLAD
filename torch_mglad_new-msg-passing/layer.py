import torch
import torch.nn as nn
import torch.nn.functional as F


class MPNNLayer(nn.Module):
    def __init__(self, num_nodes, wkr_dim, tsk_dim, num_rels, num_wkr, num_tsk):
        super(MPNNLayer, self).__init__()
        self.num_nodes = num_nodes
        self.wkr_dim = wkr_dim
        self.tsk_dim = tsk_dim
        self.num_rels = num_rels
        self.num_wkr = num_wkr
        self.num_tsk = num_tsk

        self.weight_worker = nn.Parameter(torch.Tensor(self.num_rels, self.tsk_dim, self.wkr_dim))
        self.weight_task = nn.Parameter(torch.Tensor(self.num_rels, self.wkr_dim, self.tsk_dim))

        nn.init.xavier_uniform_(self.weight_worker, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.weight_task, gain=nn.init.calculate_gain('relu'))

    def forward(self, g):
        def msg_func_wkr(edges):
            wkr_weight_type = self.weight_worker[edges.data['type'].long()]
            update_wkr_feature = torch.div(torch.matmul(edges.src['labels'], wkr_weight_type), edges.dst['deg'])
            return {'msg_ability': update_wkr_feature}

        def red_func_wkr(nodes):
            return {'ability': torch.sum(nodes.mailbox['msg_ability'], dim=1)}

        def msg_func_tsk(edges):
            tsk_weight_type = self.weight_task[edges.data['type'].long()]
            update_tsk_feature = torch.div(torch.matmul(edges.src['ability'], tsk_weight_type), edges.dst['deg'])
            return {'msg_labels': update_tsk_feature}

        def red_func_tsk(nodes):
            # return {'labels': F.sigmoid(torch.sum(nodes.mailbox['msg_labels'], dim=1))}
            return {'labels': torch.sum(nodes.mailbox['msg_labels'], dim=1)}

        # 更新 worker 的特征
        g.register_message_func(msg_func_wkr)
        g.register_reduce_func(red_func_wkr)
        g.push(g.nodes()[range(self.num_wkr, self.num_nodes)])  # push tsk 节点的msg
        g.pull(g.nodes()[range(self.num_wkr)])  # pull 到 wkr 节点

        # 更新 task 的特征
        g.register_message_func(msg_func_tsk)
        g.register_reduce_func(red_func_tsk)
        g.push(g.nodes()[range(self.num_wkr)])  # push wkr 节点的msg
        g.pull(g.nodes()[range(self.num_wkr, self.num_nodes)])  # pull 到 tsk 节点

