import torch
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
import matplotlib

#这个地方的设计现在有两个方向：
# 1.一口气更新两类节点
# 2.对两类节点分两层更新

class GladLayer(nn.Module):

    def __init__(self, wkr_feat, tsk_feat, num_rels,wkr_num, num_bases=-1, bias=None,
                 activation=None, is_input_layer=False):
        super(GladLayer, self).__init__()
        self.wkr_feat = wkr_feat
        self.tsk_feat = tsk_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.wkr_num = wkr_num
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels
        # 这个地方，bases便是边的类别，用于后面的weight的定义

        # add basis weights
        self.weight_worker = nn.Parameter(torch.Tensor(self.num_bases, self.tsk_feat,
                                                    self.wkr_feat))
        # 参数weight。有边的类别个矩阵，每个矩阵大小为[in_feat,out_feat]
        self.weight_task = nn.Parameter(torch.Tensor(self.num_bases, self.wkr_feat,
                                                self.tsk_feat))
        nn.init.xavier_uniform_(self.weight_worker,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.weight_task,
                                gain=nn.init.calculate_gain('relu'))

    # feature的选择：
    # 本部分feature让所有节点都有ability和label的feature，但是非自己部分的feature均为0。
    # 这样实现的结果就是虽然所有节点一起传播，但是只能更新到对方节点的对应feature，不属于自己的feature不会有贡献
    # 然后node还有一个feature就是出度 out degree,作为取均值的依据
    # 所以msg_func就是让ability和label均进行一次全节点的更新与传播，虽然可能浪费一点点计算量，但是并不明显。
    # 反而可能因为流程的高度统一而让传播更具效率


    def msg_func(self, edges):
        wkr_weight_type = self.weight_worker.index_select(0, torch.from_numpy(edges.data['type']).long())
        #print(self.weight_worker.shape)
        tsk_weight_type = self.weight_task.index_select(0, torch.from_numpy(edges.data['type']).long())

        edges.src['labels'][:self.wkr_num]=torch.zeros(self.wkr_num,1,self.num_rels)


        #print(wkr_weight_type.shape)
        #对权重进行选择
        #print(edges.src['labels'])
        update_wkr_feature = torch.div(torch.bmm(edges.src['labels'],(wkr_weight_type)),edges.dst['deg'])
        #这个地方给每个量除以了出度，直接相加就行

        update_tsk_feature = torch.bmm(edges.src['ability'],(tsk_weight_type))
        #print("[ *** ] update_wkr_feature_shape = ",update_wkr_feature)

        # 这个地方对每个边都采样，所以很多的起始点是一样的，值也一样
        #print("[ *** ] labels:", edges.dst['labels'])

        #print("[ *** ] ability:", edges.dst['ability'])
        return {'labels': update_tsk_feature,'ability': update_wkr_feature}

    def red_func(self,nodes):
        #print(nodes)
        # print('!!',torch.sum(nodes.mailbox['labels'],dim=1).data,torch.sum(nodes.mailbox['labels'],dim=1).data.shape)
        #print('====',F.softmax(torch.sum(nodes.mailbox['labels'], dim=1),dim=2))
        return{'labels':  F.softmax(torch.sum(nodes.mailbox['labels'], dim=1),dim=2),
               'ability': torch.sum(nodes.mailbox['ability'], dim=1)}
    def forward(self, g):
        g.update_all(message_func=self.msg_func, reduce_func=self.red_func)


class DecodeLayer(nn.Module):
    def __init__(self, wkr_feat, tsk_feat, num_rels, num_bases=-1, bias=None,
                 activation=None, is_input_layer=False):
        super(DecodeLayer, self).__init__()
        self.wkr_feat = wkr_feat
        self.tsk_feat = tsk_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        self.weight = nn.Parameter(torch.Tensor(self.wkr_feat,1))
        self.bias = nn.Parameter(torch.Tensor(1))



