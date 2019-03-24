import torch.nn as nn
from Layer import GladLayer,DecodeLayer

class mGLAD(nn.Module):
    def __init__(self, num_nodes, wkr_dim, tsk_dim, num_rels, num_bases=-1,
                 num_hidden_layers=1, dropout=0, use_cuda=False):
        super(mGLAD, self).__init__()
        self.num_nodes = num_nodes
        self.wkr_dim = wkr_dim
        self.tsk_dim = tsk_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_cuda = use_cuda

        # create rgcn layers
        self.build_model()

        # create initial features
        self.features = self.create_features()

        # TODO：获得初始feature 需要复写，并认知其作用

    def build_model(self):
        # TODO：建立模型的过程，分为input,hidden,output三部分进行，需要分别进行复现。

        # 目前思路：input，读取、建立、初始化
        #          hidden：message passing
        #          output：读取，decoder,output

        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        # for idx in range(self.num_hidden_layers):
        #     h2h = self.build_hidden_layer(idx)
        #     self.layers.append(h2h)
        # h2o

        h2h = self.build_hidden_layer()
        self.layers.append(h2h)

        h2o = self.build_output_layer()
        if h2o is not None:

            self.layers.append(h2o)

    # initialize feature for each node
    def create_features(self):
        return None

    def build_input_layer(self):
        return None

    def build_hidden_layer(self,nums=1):

        # 这个函数在建立GLAD layer，但是目前还没有对层数进行限制

        return GladLayer(wkr_feat=self.wkr_dim, tsk_feat=self.tsk_dim, num_rels=self.num_rels)

    def build_output_layer(self):
        return None

    def forward(self, g):
        if self.features is not None:
            g.ndata['id'] = self.features
        for layer in self.layers:
            layer(g)
        return g.ndata

class Decoder(nn.Module):
    def __init__(self, num_nodes, wkr_dim, tsk_dim, num_rels, num_bases=-1,
                 num_hidden_layers=1, dropout=0, use_cuda=False):
        super(Decoder, self).__init__()
        self.num_nodes = num_nodes
        self.wkr_dim = wkr_dim
        self.tsk_dim = tsk_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_cuda = use_cuda

        # create rgcn layers
        self.build_model()

        # create initial features
        self.features = self.create_features()

    def build_model(self):

        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        # for idx in range(self.num_hidden_layers):
        #     h2h = self.build_hidden_layer(idx)
        #     self.layers.append(h2h)
        # h2o

        h2h = self.build_hidden_layer()
        self.layers.append(h2h)

        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    # initialize feature for each node
    def create_features(self):
        return None

    def build_input_layer(self):
        return None

    def build_hidden_layer(self, nums=1):

        # 这个函数在建立GLAD layer，但是目前还没有对层数进行限制

        return DecodeLayer(wkr_feat=self.wkr_dim, tsk_feat=self.tsk_dim, num_rels=self.num_rels)

    def build_output_layer(self):
        return None

    def forward(self, g):
        if self.features is not None:
            g.ndata['id'] = self.features
        for layer in self.layers:
            layer(g)
        return g.ndata.pop('h')