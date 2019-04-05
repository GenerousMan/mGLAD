"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/MichSchli/RelationPrediction

Difference compared to MichSchli/RelationPrediction
* report raw metrics instead of filtered metrics
"""
import math
import argparse
import numpy as np
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from dgl.contrib.data import load_data

from Model import mGLAD
import numpy as np
import matplotlib.pyplot as plt
from utils import *


# 要注意，本部分代码的设置如下：
#  1.输入的边信息是双向的，所有wkr,tsk都有两向连接关系，
#  2.要有一个矩阵记录原始的tsk,wkr边信息（即非双向），以供loss计算查找

# TODO：
#  1.加上softmax和除以均值。
#  2.loss计算负号。
#  3.输出embbeding预测和边准确度预测


class GLADLinkPredict(nn.Module):

    # 这个地方应该就是咱们最后的模型
    # 要修改的点：
    # TODO 1：修改模型结构，把rgcn改为mGLAD,查明messagefunction 的传播机制，看是否能有
    #  判断、不同的传播机制。现在的想法是判断是否存在这类成员data，有与没有采取不同的传播、reduce函数
    #  其中的编码层多加一个参数，多写一种update function（其实就是多一类节点的判断）
    # TODO 2: 改变中间的层结构，将latent variable赋予真实意义，即worker['label']里的message必须有存在意义
    # TODO 3: 改变loss定义函数。将edge 的类别算入考虑

    def __init__(self, num_nodes, wkr_num, wkr_dim, tsk_dim, num_rels, num_bases=-1,
                 num_hidden_layers=1, dropout=0, use_cuda=False, reg_param=0):
        super(GLADLinkPredict, self).__init__()
        self.wkr_dim = wkr_dim
        self.tsk_dim = tsk_dim
        self.wkr_num = wkr_num
        self.num_rels = num_rels
        self.mGLAD = mGLAD(num_nodes=num_nodes, wkr_num=wkr_num, wkr_dim=self.wkr_dim, tsk_dim=self.tsk_dim,
                           num_rels=self.num_rels)

        self.reg_param = reg_param
        self.w_relation = nn.Parameter(torch.Tensor(wkr_dim, 1))
        self.bias = nn.Parameter(torch.Tensor(1,1))
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.bias, gain=nn.init.calculate_gain('relu'))




    def calc_score(self, nodes, triplets):
        # DistMult
        # 这个地方的triplets不能有两向边，只能单向

        # print("[ *** ] now triplets' shape:", triplets)
        relation = triplets[:, 1]  # numpy.ndarray
        # 所有的relations关系，取出所有的边即可

        tsk = triplets[:, 2]
        # 所有的tsk节点编号，是一个list，将其ability取出
        # 之后要取出里面所有指定的label feature

        wkr = triplets[:, 0]
        # print(wkr.shape)
        # 所有的wkr节点编号，一个list，
        # 将所有ability feature取出
        # print(nodes)
        nodes['labels'] = F.softmax(nodes['labels'], dim=2)
        #print(nodes['labels'])

        wkr_feature = nodes['ability'][torch.from_numpy(wkr.astype(int)).long()]
        # print("[ *** ] worker 's features now we choose is:", wkr_feature)
        # print(wkr_feature.shape)
        # tsk_feature = nodes['labels'][torch.from_numpy(tsk.astype(int)).long(), :, relation]
        tsk_feature = nodes['labels'][torch.from_numpy(tsk.astype(int)).long()]
        #print('tau',tsk_feature)
        # print("[ *** ] task 's features  now are:", nodes['labels'][torch.from_numpy(tsk.astype(int)).long(),:,relation].data)

        score_part1 = torch.sigmoid((torch.matmul(wkr_feature, self.w_relation) + self.bias))

        # print("[ *** ] wkr feature is: ",nodes['ability'])
        # print("[ *** ] after choosing:",wkr_feature)
        # print("[ *** ] w relation is: ", self.w_relation)
        # print("[ *** ] before sigmoid:",(torch.matmul(wkr_feature,self.w_relation)+self.bias))
        score_part2 = (1 - score_part1) / (self.num_rels - 1)
        # print('score_part1',score_part1.shape,score_part1)  #[4212,1,1]
        # print('tsk_feature',tsk_feature.shape,tsk_feature)  #[4212,1,2]

        score = torch.matmul(score_part1,tsk_feature)+torch.matmul(score_part2,(1-tsk_feature))
        #print("[ *** ] score's shape:", score)
        # score = torch.pow(score_part1, tsk_feature) + torch.pow(score_part2, 1 - tsk_feature)

        # TODO: 要检查本部分代码正确性。因为是批处理，所以需要小心维度等信息

        return score





    def forward(self, g):
        return self.mGLAD.forward(g)

    def evaluate(self, g):
        # get embedding and relation weight without grad
        embedding = self.forward(g)
        return embedding, self.w_relation

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def get_loss(self, g, triplets):
        #TODO 这边还有问题，把所有p(y_ji)都加上了，应该改成 p(y_ji=l)
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        embedding = self.forward(g)
        # score = self.calc_score(embedding, triplets)
        # print(score.shape)
        # print(score)
        # predict_loss = -1 * torch.sum(torch.log(score))
        # print(predict_loss)
        relation=triplets[:,1]
        relation.shape=[-1,1]
        #print(relation.shape)
        score = self.calc_score(embedding,triplets)
        predict_edge=np.argmax(score.detach().numpy(),axis=2)
        #print(predict_edge.shape)
        predict_edge_acc=np.equal(predict_edge,relation).sum()/predict_edge.shape[0]
        print("[ data ] Predict edge's accuracy: ",predict_edge_acc)

        score=torch.squeeze(score,1)
        mask=torch.zeros(len(triplets),self.num_rels).scatter_(1, torch.LongTensor(relation), 1)
        #print(mask.shape)

        #score=score[:,0,relation]
        #print(score)
        #mask=torch.unsqueeze(mask, 1)
        #print(mask)
        #print(score)
        #print(score*mask)
        predict_loss = -1 * torch.sum(torch.log(score).mul(mask))
        #print('predict_loss',predict_loss.shape,predict_loss)  #[4212,1,2]

        # reg_loss = self.regularization_loss(embedding)
        return embedding, predict_loss,predict_edge_acc


def draw(edge_acc, acc, loss, name):
    # loss: numpy

    x = range(len(acc))
    y1 = acc
    y2 = loss
    y3 = edge_acc
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(x, y1, label='label acc')
    ax2 = fig.add_subplot(212)
    ax2.plot(x, y2, label='loss')
    ax1.plot(x, y3, label='edge acc')

    #plt.show()
    plt.savefig(name+".png")


def main(args):
    # load graph data
    data, num_nodes, num_rels, wkr_num, true_labels, name = read_Websites()
    # TODO: 数据集部分需要修改，
    #  此处数据集的形状： [ src node, type, dest node ]。
    #  基本上就是[ worker, type, task ]，
    #  shape: [edge's num, 3]
    #  然后所有的节点都得反过来存一遍。

    train_data = data
    # print("[ *** ] training data:", train_data.shape)
    valid_data = data
    # print("[ *** ] validating data:", valid_data.shape)
    test_data = data
    # print("[ *** ] testing data:", test_data)

    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)

    # create model
    model = GLADLinkPredict(num_nodes=num_nodes,
                            wkr_dim=args.n_hidden,
                            tsk_dim=num_rels,
                            wkr_num=wkr_num,
                            num_rels=num_rels,
                            num_bases=args.n_bases,
                            num_hidden_layers=args.n_layers,
                            dropout=args.dropout,
                            use_cuda=use_cuda,
                            reg_param=args.regularization)

    # validation and testing triplets
    valid_data = torch.LongTensor(valid_data)
    test_data = torch.LongTensor(test_data)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model_state_file = name+".pth"#'model_state.pth'
    forward_time = []
    backward_time = []

    # training loop
    print("start training...")

    epoch = 0
    best_mrr = 0
    transverse_data = data[:, [2, 1, 0]]
    triplets_before = np.concatenate((data, transverse_data), axis=0)
    triplets = (data[:, 0], data[:, 1], data[:, 2])
    loss_all = []
    acc_all = []
    edge_all=[]

    # print("[ *** ]triplets: ", triplets)
    if use_cuda:
        model.cuda()

    if(os.path.isfile(name+".pkl")):
        model.load_state_dict(torch.load(name+".pkl"))
        print("[ model ]model loaded")



    while True:
        model.train()
        epoch += 1
        g, rel, _ = build_graph_from_triplets(num_nodes, num_rels, triplets)
        # print("[ *** ] Shape of triplets_before:",triplets_before.shape)
        g.edata['type'] = triplets_before[:, 1].astype(int)

        g.ndata['labels'] = torch.zeros(num_nodes, 1, num_rels)
        g.ndata['ability'] = torch.zeros(num_nodes, 1, args.n_hidden)
        g.ndata['deg'] = g.out_degrees(g.nodes()).float().view(-1, 1, 1)
        # g.nodes[range(wkr_num)].data['labels'] = torch.zeros(wkr_num, 1, num_rels)
        # g.nodes[range(wkr_num, num_nodes)].data['labels'] = torch.rand(num_nodes - wkr_num, 1, num_rels)
        #
        # g.nodes[range(wkr_num)].data['ability'] = torch.rand(wkr_num, 1, args.n_hidden)
        # g.nodes[range(wkr_num, num_nodes)].data['ability'] = torch.zeros(num_nodes - wkr_num, 1, args.n_hidden)

        # 这里改成初始值相同
        g.nodes[range(wkr_num)].data['labels'] = torch.zeros(wkr_num, 1, num_rels)
        g.nodes[range(wkr_num, num_nodes)].data['labels'] = torch.ones(num_nodes - wkr_num, 1, num_rels)

        g.nodes[range(wkr_num)].data['ability'] = torch.ones(wkr_num, 1, args.n_hidden)
        g.nodes[range(wkr_num, num_nodes)].data['ability'] = torch.zeros(num_nodes - wkr_num, 1, args.n_hidden)

        # 这个图应该就是普通的DGLGraph
        # 这个地方进行了采样，但是edge的量都极大，我们应该暂时不需要采样。
        # 本部分掠过
        t0 = time.time()

        embedding, loss, edge_acc = model.get_loss(g, data)
        # 这个地方的data是单向图，并不存在反向边
        predict_label = np.argmin(embedding['labels'][range(wkr_num, num_nodes)].detach().numpy(), axis=2)
        # print("[ data ] Each Wkr features: ",embedding['ability'])
        #print("[ data ] Each Tsk features: ", embedding['labels'])
        predict_label.shape = num_nodes - wkr_num
        true_labels.shape = num_nodes - wkr_num
        mrr = np.sum(np.equal(predict_label, true_labels)) / (num_nodes - wkr_num)
        acc_all.append(mrr)
        loss_all.append(loss)
        edge_all.append(edge_acc)

        print("[ data ] Predict label's accuracy: ",mrr)

        if mrr > best_mrr:
            best_mrr = mrr
        t1 = time.time()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
        optimizer.step()
        t2 = time.time()
        forward_time.append(t1 - t0)
        backward_time.append(t2 - t1)
        print("Epoch {:04d} | Loss {:.4f} | Best MRR {:.4f} | Forward {:.4f}s | Backward {:.4f}s".
              format(epoch, loss.item(), best_mrr, forward_time[-1], backward_time[-1]))

        # print('worker-label')
        # print(g.nodes[range(wkr_num)].data['labels'])
        # print('task-label')
        # print(g.nodes[range(wkr_num, num_nodes)].data['labels'])
        #
        # print('worker-ability')
        # print(g.nodes[range(wkr_num)].data['ability'])
        # print('task-ability')
        # print(g.nodes[range(wkr_num, num_nodes)].data['ability'])

        optimizer.zero_grad()

        if epoch % 200 ==0 :
            torch.save(model.state_dict(), name+".pkl")#draw(np.array(acc_all),np.array(loss_all))
            print("[ model ] model saved.")
            draw(np.array(edge_all),np.array(acc_all), np.array(loss_all),name)
            print("[ data ] draw finished.")


        if epoch >= args.n_epochs:
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                       model_state_file)
            break

    print("training done")
    # print("Mean forward time: {:4f}s".format(np.mean(forward_time)))
    # print("Mean Backward time: {:4f}s".format(np.mean(backward_time)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mGLAD')
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=300,  # !
                        help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-5,  # lr
                        help="learning rate")
    parser.add_argument("--n-bases", type=int, default=200,
                        help="number of weight blocks for each relation")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--n-epochs", type=int, default=10000,
                        help="number of minimum training epochs")
    parser.add_argument("-d", "--dataset", type=str, required=False,
                        help="dataset to use")
    parser.add_argument("--eval-batch-size", type=int, default=500,
                        help="batch size when evaluating")
    parser.add_argument("--regularization", type=float, default=0.01,
                        help="regularization weight")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")
    # parser.add_argument("--graph-batch-size", type=int, default=30000,
    #                     help="number of edges to sample in each iteration")
    # parser.add_argument("--graph-split-size", type=float, default=0.5,
    #                     help="portion of edges used as positive sample")
    # parser.add_argument("--negative-sample", type=int, default=10,
    #                     help="number of negative samples per positive sample")
    # parser.add_argument("--evaluate-every", type=int, default=500,
    #                     help="perform evaluation every n epochs")

    args = parser.parse_args()
    print(args)
    main(args)
