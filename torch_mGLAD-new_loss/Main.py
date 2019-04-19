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

class GLADLinkPredict(nn.Module):

    def __init__(self, num_nodes, wkr_num, wkr_dim, tsk_dim, num_rels, r,edges,
                 num_hidden_layers=1, dropout=0, use_cuda=False, reg_param=0):
        super(GLADLinkPredict, self).__init__()
        self.wkr_dim = wkr_dim
        self.tsk_dim = tsk_dim
        self.wkr_num = wkr_num
        self.num_rels = num_rels
        self.mGLAD = mGLAD(num_nodes=num_nodes, wkr_num=wkr_num, wkr_dim=self.wkr_dim, tsk_dim=self.tsk_dim,
                           num_rels=self.num_rels)
        self.r = r

        self.reg_param = reg_param

        self.w_relation = nn.Parameter(torch.Tensor(wkr_dim, num_rels))  # D*L
        self.bias = nn.Parameter(torch.Tensor(1, num_rels))  # 1*L

        self.pi = nn.Parameter(torch.Tensor(num_rels, num_rels))
        self.edges=edges
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.bias, gain=nn.init.calculate_gain('relu'))

        nn.init.xavier_uniform_(self.pi, gain=nn.init.calculate_gain('relu'))

    def calc_score(self, nodes, triplets):

        relation = triplets[:, 1]  # numpy.ndarray
        tsk = triplets[:, 2]
        wkr = triplets[:, 0]
        rel = triplets[:, 1]
        rel=rel.astype(np.int)


        nodes['labels'] = F.softmax(nodes['labels'], dim=2)

        wkr_feature = nodes['ability'][torch.from_numpy(wkr.astype(int)).long()].squeeze()    # [4212,1,300]  a_j

        tsk_feature = nodes['labels'][torch.from_numpy(tsk.astype(int)).long()]   # tau_i
        #print("[ *** ] task 's features  now are:", nodes['labels'][torch.from_numpy(tsk.astype(int)).long(),:,relation].data)
        tsk_feature=tsk_feature.squeeze()
        P=torch.zeros(self.edges,self.num_rels)
        aw_b = torch.matmul(wkr_feature, self.w_relation) + self.bias
        aw_b=F.softmax(aw_b)
        #pi_all=self.pi[range(self.edges),rel]
        print(tsk_feature.shape)
        print(aw_b.shape)

        #按公式写的，已经对pi作了softmax
        score=(tsk_feature*aw_b).matmul(F.softmax(self.pi,dim=1))

        print("pi_now:",F.softmax(self.pi,dim=0))
        print("[ *** ] score 's shape:",score.shape)

        return score

    def forward(self, g):
        return self.mGLAD.forward(g)

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def get_loss(self, g, triplets):

        embedding = self.forward(g)

        relation = triplets[:, 1]
        tsk = triplets[:, 2]

        relation_one_hot = torch.zeros(self.edges, self.num_rels)
        relation_one_hot[range(self.edges), relation] = 1
        # one hot matrics


        relation.shape = [-1,1]

        score = self.calc_score(embedding, triplets)
        #print(score)
        predict_edge = np.argmax(score.detach().numpy(), axis=1)

        print("predict edges:",predict_edge)
        print("original relation:",relation)
        predict_edge_acc = np.equal(predict_edge, relation.squeeze()).sum() / predict_edge.shape[0]

        print("[ data ] Predict edge's accuracy: ", predict_edge_acc)

        loss = nn.CrossEntropyLoss()

        predict_loss = loss(score,torch.tensor(relation).squeeze().long())
        #交叉熵的计算，但是这个地方不需要one hot

        #predict_loss = -1 * torch.sum(torch.log(score.squeeze()[range(self.edges),relation.squeeze()]))
        print("The target scores:",score.squeeze()[range(self.edges),relation.squeeze()].shape)

        reg_loss = self.regularization_loss(embedding['labels'])

        entropy = nn.CrossEntropyLoss()
        kl_loss = entropy(embedding['labels'][self.wkr_num:], torch.LongTensor(self.r))

        loss = predict_loss# + 0.005 * reg_loss + 0.0001 * kl_loss

        # reg_loss = self.regularization_loss(embedding)
        return embedding, loss, predict_edge_acc
        # return embedding, loss


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

    # plt.show()
    plt.savefig(name + ".png")


def main(args):
    # load graph data
    data, num_nodes, num_rels, wkr_num, true_labels, name, r = read_BlueBirds()
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
    print("data's lengtg:",len(data))
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
                            num_hidden_layers=args.n_layers,
                            dropout=args.dropout,
                            use_cuda=use_cuda,
                            reg_param=args.regularization,
                            r=r,
                            edges=len(data))

    # validation and testing triplets
    valid_data = torch.LongTensor(valid_data)
    test_data = torch.LongTensor(test_data)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model_state_file = name + ".pth"  # 'model_state.pth'
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
    edge_all = []

    # print("[ *** ]triplets: ", triplets)
    if use_cuda:
        model.cuda()

    # if (os.path.isfile(name + ".pkl")):
    #     model.load_state_dict(torch.load(name + ".pkl"))
    #     print("[ model ]model loaded")

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
        # embedding, loss = model.get_loss(g, data)
        # 这个地方的data是单向图，并不存在反向边
        # print('labels',embedding['labels'][range(wkr_num, num_nodes)].detach().numpy())
        predict_label = np.argmax(embedding['labels'][range(wkr_num, num_nodes)].detach().numpy(), axis=2)
        # print('predict_label',predict_label)
        # print("[ data ] Each Wkr features: ",embedding['ability'])
        # print("[ data ] Each Tsk features: ", embedding['labels'])
        predict_label.shape = num_nodes - wkr_num
        true_labels.shape = num_nodes - wkr_num
        mrr = np.sum(np.equal(predict_label, true_labels)) / (num_nodes - wkr_num)
        acc_all.append(mrr)
        loss_all.append(loss)
        edge_all.append(edge_acc)

        print("[ data ] Predict label's accuracy: ", mrr)

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

        if epoch % 20 == 0:
            torch.save(model.state_dict(), name + ".pkl")  # draw(np.array(acc_all),np.array(loss_all))
            print("[ model ] model saved.")
            draw(np.array(edge_all), np.array(acc_all), np.array(loss_all), name)
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
    parser.add_argument("--lr", type=float, default=1e-4,  # lr
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
