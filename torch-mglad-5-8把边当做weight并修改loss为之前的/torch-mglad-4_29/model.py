import torch.nn as nn
from layer import MGLADlayer
import torch


class MGLAD(nn.Module):
    def __init__(self, num_nodes, num_wkr, num_tsk, wkr_dim, tsk_dim, num_rels, num_layers=1):
        super(MGLAD, self).__init__()
        self.num_nodes = num_nodes
        self.num_wkr = num_wkr
        self.num_tsk = num_tsk
        self.wkr_dim = wkr_dim
        self.tsk_dim = tsk_dim
        self.num_rels = num_rels
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        self.build_model()

    def build_layer(self):
        return MGLADlayer(wkr_dim=self.wkr_dim, num_nodes=self.num_nodes, num_wkr=self.num_wkr, num_tsk=self.num_tsk,
                          tsk_dim=self.tsk_dim,
                          num_rels=self.num_rels)

    def build_model(self):
        for _ in range(self.num_layers):
            layer = self.build_layer()
            self.layers.append(layer)

    def forward(self, g):
        for layer in self.layers:
            layer(g)
        return g
