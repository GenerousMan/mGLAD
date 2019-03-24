import networkx as nx
import matplotlib.pyplot as plt
import torch
import dgl
from utils import *


shape, wkr_num, tsk_num, edge_type, graph, true_labels=read_duck()
print(graph)

def graph2dgl(graph):
    g = dgl.DGLGraph()
    # 创建图
    # 添加节点，共有wkr_num+tsk_num个节点。
    # 给节点添加边，边的关系即为评判的结果
    #