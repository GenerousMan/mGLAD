# -*- coding: utf-8 -*-
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import tensorflow as tf

from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import yaml


def Cal_ProbLoss(loss, P, edges):
    # predict_edges' shape:(K*x)
    
    def cond_worker(i,loss_now):
        #判断第i个worker

        return i < edges.shape[0]
    def body_worker(i,loss_now):
        # 对loss进行累加运算
        def cond_task(j,loss_now):
            # 判断第j个task

            return j<edges.shape[1]

        def body_task(j,loss_now):

            # 对loss进行累加运算
            loss = tf.add(loss,P[i][j][edges[i][j]])

            return j+1, loss 
        loss=tf.while_loop(cond_task,body_task,[0,loss])

        return i+1, loss 

    loss=tf.while_loop(cond_worker,body_worker,[0,loss])

    return loss
def Graph2Edgelist(Graph):
    workers_num = Graph.shape[0]
    tasks_num = Graph.shape[1]
    edge_list=[]
    task_num_for_wkr=[]

    for worker in range(workers_num):
        count=0
        for task in range(tasks_num):
            if(Graph[worker][task]==-1):
                continue
            else:
                count+=1
                edge_list.append([worker,task,Graph[worker][task]])
        task_num_for_wkr.append([count])
    tsk_num_for_wkr=np.array(task_num_for_wkr)
    tsk_num_for_wkr.shape=[tsk_num_for_wkr.shape[0]]
    return np.array(edge_list),tsk_num_for_wkr

def read_duck():
    print("[ data ] Now reading the Duck dataset...")
    f=open("./ducks/labels.txt")
    ducks_info=f.readlines()
    workers_num=40
    tasks_num=240
    edges_type=2

    f2=open("./ducks/map.yaml")
    gtLabels = yaml.load(f2)
    workerId=gtLabels['wkr']
    wkr_num=len(workerId.keys())
    workerId2Idx=dict((id, idx) for (idx, id) in enumerate(workerId.values()))
    print(workerId2Idx)
    graph=np.zeros(wkr_num*240)
    graph.shape=[wkr_num,240]
    graph-=1
    #count_wkr=np.zeros(60)
    print("[ data ] Building the Graph......")
    for i in range(len(ducks_info)):
        x = ducks_info[i]
        #print(x)
        x = str(x).split("\n")[0]#.split[" "]]
        x = [int(j) for j in str(x).split(" ")]
        #count_wkr[x[1]]+=1
        graph[workerId2Idx[x[1]]-1][x[0]-1]=x[2]

        #print(x)
    print("[ data ] Graph built finished.")
    f3=open("./ducks/classes.yaml")
    Labels = yaml.load(f3)
    true_labels = np.zeros(240)
    for i in range(240):
        true_labels[i] = (int(Labels['labels'][i])!=3)

    print("[ data ]  Now getting the true labels.")

    return graph.shape, wkr_num, 240, 2, graph, true_labels

def read_BlueBirds():
    # 构建整幅图，39*104
    # 返回图的邻接矩阵

    print("[ data ] Now loading blueBirds dataset......")
    numTrial=40
    f=open("./bluebirds/gt.yaml")
    gtLabels = yaml.load(f)
    imgIds = gtLabels.keys(); numImg = len(gtLabels)
    imgId2Idx = dict((idx, id) for (idx, id) in enumerate(imgIds))
    print(imgId2Idx)
    data = yaml.load(open("./bluebirds/labels.yaml"))
    dinfo = { 'numImg' : numImg, 'numTrial' : numTrial }
    dinfo['gt'] = [gtLabels[id] for id in imgIds]

    wkrIds = data.keys();
    wkrId2Idx = dict((idx, id) for (idx, id) in enumerate(wkrIds))
    print("[ data ] Dataset has ",len(wkrIds)," woker nodes, ",len(imgIds),"task nodes.")
    print("[ data ] Now building the original graph......")
    Graph=np.zeros((len(wkrIds),len(imgIds)))
    for i in range(len(wkrIds)):
        for j in range(len(imgIds)):
            #print(data[wkrId2Idx[i]][imgId2Idx[j]])
            Graph[i][j]=int(data[wkrId2Idx[i]][imgId2Idx[j]])
    print("[ data ] Build Graph Finished. ")
    print("[ data ] Now get the true labels...")
    TrueLabels=np.zeros(len(imgIds))
    for i in range(len(imgIds)):
        TrueLabels[i]=gtLabels[imgId2Idx[i]]

    return Graph.shape,len(wkrIds),len(imgIds),2,Graph,TrueLabels

def read_Websites():

    print("[ data ] Now loading the Website dataset......")
    filename = "./web_processed_data_feature_2"
    # filename = "age_data_3_category"
    # filename = "bluebird_data"
    data_all = np.load(filename + '.npz')
    user_labels = data_all['user_labels']
    #print(user_labels[:,5:10])
    label_mask = data_all['label_mask']
    #print(label_mask)
    true_labels = data_all['true_labels']

    category_size = data_all['category_num']
    source_num = data_all['source_num']
    n_samples, _ = np.shape(true_labels)
    #return Graph.shape,len(wkrIds),len(imgIds),2,Graph,TrueLabels
    Graph=np.zeros(source_num*n_samples)-1
    Graph.shape=[source_num,n_samples]
    true_labels.shape=n_samples
    print("[ data] Now building the Graph.......")
    for i in range(source_num):
        for j in range(n_samples):
            for k in range(5):
                if(user_labels[j][5*i+k]==1):
                    Graph[i][j]=k
    print(Graph.shape)
    print("[ data ] Building finished.")
    return Graph.shape,source_num,n_samples,5,Graph,true_labels
def read_Flowers():

    print("[ data ] Now loading the Flower dataset......")
    filename = "./flower_data"
    # filename = "age_data_3_category"
    # filename = "bluebird_data"
    data_all = np.load(filename + '.npz')
    user_labels = data_all['user_labels']
    print(user_labels.shape)
    label_mask = data_all['label_mask']
    #print(label_mask)
    true_labels = data_all['true_labels']
    print(true_labels)
    category_size = data_all['category_num']
    source_num = data_all['source_num']
    n_samples, _ = np.shape(true_labels)

    #return Graph.shape,len(wkrIds),len(imgIds),2,Graph,TrueLabels
    Graph=np.zeros(source_num*n_samples)-1
    Graph.shape=[source_num,n_samples]
    true_labels.shape=n_samples
    print("[ data] Now building the Graph.......")
    for i in range(source_num):
        for j in range(n_samples):
            for k in range(2):
                if(user_labels[j][2*i+k]==1):
                    Graph[i][j]=k
    #print(Graph[7])
    print("[ data ] Building finished.")
    return Graph.shape,source_num,n_samples,2,Graph,true_labels

def read_Ducks():
    print("[ data ] Now loading Duck dataset......")
    f=open("../ducks/labels.txt")
    ducks_info=f.readlines()
    print(ducks_info)
    return 

def construct_feed_dict(edges,worker_num,task_num,edge_type,ability_num,placeholders,tsk_num_for_wkr,graph):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['edges']: edges})
    feed_dict.update({placeholders['graph']: graph})
    feed_dict.update({placeholders['worker_num']: worker_num})
    feed_dict.update({placeholders['task_num']: task_num})
    feed_dict.update({placeholders['edge_type']: edge_type})
    feed_dict.update({placeholders['ability_num']: ability_num})
    feed_dict.update({placeholders['tsk_num_for_wkr']: tsk_num_for_wkr})
    #feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)



def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)
