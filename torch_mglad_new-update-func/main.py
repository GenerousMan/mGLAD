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

from model import MGLAD
import numpy as np
import matplotlib.pyplot as plt
from utils import *


class GLADLinkPredict(nn.Module):
    def __init__(self, num_nodes, num_wkr, num_tsk, num_rels, wkr_dim, tsk_dim, e_dim,mv_results, num_layers, edges,
                 dropout=0,
                 use_cuda=False, reg_param=0):
        super(GLADLinkPredict, self).__init__()
        self.num_nodes = num_nodes
        self.num_wkr = num_wkr
        self.num_tsk = num_tsk
        self.wkr_dim = wkr_dim
        self.tsk_dim = tsk_dim
        self.e_dim = e_dim
        self.num_rels = num_rels
        self.mv_results = mv_results
        self.edges = edges


        self.weight = nn.Parameter(torch.Tensor(wkr_dim, num_rels))  # D*L
        self.bias = nn.Parameter(torch.Tensor(1, num_rels))  # 1*L
        self.pi = nn.Parameter(torch.Tensor(num_rels, num_rels))  # L*L
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.bias, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.pi, gain=nn.init.calculate_gain('relu'))

        # self.weight_1 = nn.Parameter(torch.Tensor(wkr_dim+num_rels, num_rels))  # D*L
        # self.bias_1 = nn.Parameter(torch.Tensor(1, num_rels))  # 1*L
        # nn.init.xavier_uniform_(self.weight_1, gain=nn.init.calculate_gain('relu'))
        # nn.init.xavier_uniform_(self.bias_1, gain=nn.init.calculate_gain('relu'))

        # 两层的mpnn
        self.mglad = MGLAD(num_nodes=num_nodes, num_wkr=num_wkr, num_tsk=num_tsk, wkr_dim=wkr_dim, tsk_dim=tsk_dim,
                           num_rels=num_rels, e_dim=e_dim,num_layers=num_layers)

    def cal_score(self, ndata_, triplets):
        # triplets 单向边
        # [wkr, rel, tsk]
        wkr = triplets[:, 0]
        rel = triplets[:, 1].astype(np.int)
        tsk = triplets[:, 2]

        # 这里对tau做softmax
        ndata_['labels'] = F.softmax(ndata_['labels'], dim=2)
        print('labels:',ndata_['labels'])

        wkr_feature = ndata_['ability'][torch.from_numpy(wkr.astype(int)).long()].squeeze()  # [4212,1,300]  a_j
        tsk_feature = ndata_['labels'][torch.from_numpy(tsk.astype(int)).long()].squeeze()  # tau_i

        aw_b = F.softmax(torch.matmul(wkr_feature, self.weight) + self.bias, dim=1)

        # 在使用pi之前对其做softmax
        # score = (tsk_feature.mul(aw_b)).matmul(F.softmax(self.pi, dim=1))
        score = (tsk_feature.mul(aw_b))#.matmul(F.softmax(self.pi, dim=1))

        # a_cat_tau = torch.cat((wkr_feature,tsk_feature),dim=1)
        # score = F.softmax(F.sigmoid(a_cat_tau.matmul(self.weight_1)+self.bias_1),dim=1)

        # print("pi_now:", F.softmax(self.pi, dim=1))

        # 使p(y_ji)几种label的值相加为1
        # score = F.softmax(score, dim=1)  # E * L, p(y_ji)  [4212,2]  貌似使用crossentropy不需要进行softmax
        return score

    def forward(self, g):
        return self.mglad.forward(g)

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.weight.pow(2))

    def get_loss(self, g, triplets):
        # triplets 单向边
        # [wkr, rel, tsk]
        g = self.forward(g)
        ndata_ = g.ndata

        rel = triplets[:, 1]
        rel.shape = [-1, 1]
        score = self.cal_score(ndata_, triplets)
        predict_edges = torch.argmax(score, dim=1)  #[4212,]
        predict_edge_acc = np.equal(predict_edges.detach().numpy(), rel.squeeze()).sum() / predict_edges.shape[0]

        print("predict edges:", predict_edges)
        print("[ data ] Predict edge's accuracy: ", predict_edge_acc)

        loss = nn.CrossEntropyLoss()  # CrossEntropyLoss() 是 softmax 和 负对数损失的结合
        predict_loss = loss(score, torch.tensor(rel).squeeze().long())

        reg_loss = self.regularization_loss(ndata_['labels'])

        kl_loss = loss(ndata_['labels'][self.num_wkr:], torch.LongTensor(self.mv_results))

        loss_sum = predict_loss  # + 0.005 * reg_loss + 0.0001 * kl_loss


        return g, loss_sum, predict_edge_acc

# --------------------------------------------------------------------------------------------------------

def draw(edge_acc, acc, loss, dataset_name):
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
    plt.savefig(dataset_name + ".png")


def main(args):
    # data, num_nodes, num_rels, num_wkr, num_tsk, true_labels, name, mv_results = read_BlueBirds()
    # data, num_nodes, num_rels, num_wkr, num_tsk, true_labels, name, mv_results = read_Flowers()
    data, num_nodes, num_rels, num_wkr, num_tsk, true_labels, name, mv_results = read_Websites()
    # data, num_nodes, num_rels, num_wkr, num_tsk, true_labels, name, mv_results = read_duck()
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)

    # create model
    model = GLADLinkPredict(num_nodes=num_nodes,
                            num_wkr=num_wkr,
                            num_tsk=num_tsk,
                            num_rels=num_rels,
                            wkr_dim=args.wkr_dim,
                            tsk_dim=num_rels,
                            e_dim=args.e_dim,
                            mv_results=mv_results,
                            num_layers=args.n_layers,
                            edges=len(data),
                            dropout=args.dropout,
                            use_cuda=use_cuda,
                            reg_param=args.regularization
                            )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model_state_file = name + ".pth"
    forward_time = []
    backward_time = []

    print("start training...")

    # 画图
    loss_all = []
    acc_all = []
    edge_all = []

    # triplets
    # data [wkr, rel, tsk]
    # transverse_data [tsk, rel, wkr]
    # triplets_before 双向边
    # triplets (array([ 0.,  0.,  0., ..., 38., 38., 38.]), array([1., 1., 1., ..., 0., 0., 0.]), array([ 39.,  40.,  41., ..., 144., 145., 146.]))
    transverse_data = data[:, [2, 1, 0]]
    triplets_before = np.concatenate((data, transverse_data), axis=0)
    triplets = (data[:, 0], data[:, 1], data[:, 2])

    epoch = 0
    best_mrr = 0

    if use_cuda:
        model.cuda()

    while True:
        model.train()
        g, rel, _ = build_graph_from_triplets(num_nodes, num_rels, triplets)
        g.edata['type'] = torch.tensor(triplets_before[:, 1].astype(int))
        g.ndata['labels'] = torch.zeros(num_nodes, 1, num_rels)
        g.ndata['ability'] = torch.zeros(num_nodes, 1, args.wkr_dim)
        g.ndata['deg'] = g.out_degrees(g.nodes()).float().view(-1, 1, 1)

        g.nodes[range(num_wkr)].data['ability'] = torch.rand(num_wkr, 1, args.wkr_dim)
        g.nodes[range(num_wkr, num_nodes)].data['labels'] = torch.rand(num_nodes - num_wkr, 1, num_rels)

        t0 = time.time()
        g, loss, edge_acc = model.get_loss(g, data)
        predict_labels = np.argmax(g.ndata['labels'][range(num_wkr, num_nodes)].detach().numpy(), axis=2)
        print('predict labels:',predict_labels.transpose())
        predict_labels.shape = num_nodes - num_wkr
        true_labels.shape = num_nodes - num_wkr

        mrr = np.sum(np.equal(predict_labels, true_labels)) / num_tsk

        acc_all.append(mrr)
        loss_all.append(loss)
        edge_all.append(edge_acc)
        print("[ data ] Predict label's accuracy: ", mrr)
        if mrr > best_mrr:
            best_mrr = mrr

        t1 = time.time()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
        optimizer.step()
        t2 = time.time()
        forward_time.append(t1 - t0)
        backward_time.append(t2 - t1)
        print("Epoch {:04d} | Loss {:.4f} | Best MRR {:.4f} | Forward {:.4f}s | Backward {:.4f}s".
              format(epoch, loss.item(), best_mrr, forward_time[-1], backward_time[-1]))
        optimizer.zero_grad()
        if epoch % 20 == 0:
            # torch.save(model.state_dict(), name + ".pkl")  # draw(np.array(acc_all),np.array(loss_all))
            # print("[ model ] model saved.")
            draw(np.array(edge_all), np.array(acc_all), np.array(loss_all), name)
            print("[ data ] draw finished.")

        if epoch >= args.n_epochs:
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                       model_state_file)
            break

        epoch += 1

    print("training done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mGLAD')
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="dropout probability")
    parser.add_argument("--wkr-dim", type=int, default=100,  # !
                        help="dimension of worker ability")
    parser.add_argument("--e-dim", type=int, default=10,  # !
                        help="dimension of e_l (a vector to represent edge with label l)")
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
