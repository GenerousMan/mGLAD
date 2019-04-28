# -*- coding: utf-8 -*-
import numpy as np
import yaml
import dgl


def Graph2Edgelist(Graph):
    workers_num = Graph.shape[0]
    tasks_num = Graph.shape[1]
    edge_list = []
    for worker in range(workers_num):
        count = 0
        for task in range(tasks_num):
            if (Graph[worker][task] == -1):
                continue
            else:
                count += 1
                edge_list.append([worker, Graph[worker][task], workers_num + task])
                # 把tsk节点存在wkr节点之后，编号顺序
    return np.array(edge_list)


def Graph2r(edges, tsk_num, num_rels, wkr_num):
    # 计算 majority voting的结果
    mv_results = np.zeros((tsk_num, num_rels))
    for i in range(tsk_num):
        label_i = list(edges[:, i])
        label_il = {}
        for l in range(num_rels):
            label_il[l] = label_i.count(float(l))
            mv_results[i][l] = label_il[l] / wkr_num
        # 第i个task的label l worker 投票的概率
        # 这里其实不应该直接除以worker数，
        # 但在bluebird这样的所有worker对所有task投票的数据集上可以用，
        # 此处不太重要
    return mv_results


def read_BlueBirds():
    # 构建整幅图，39*104
    # 返回图的邻接矩阵
    print("[ data ] Now loading blueBirds dataset......")
    numTrial = 40
    f = open("./bluebirds/gt.yaml")
    gtLabels = yaml.load(f)
    imgIds = gtLabels.keys();
    numImg = len(gtLabels)
    imgId2Idx = dict((idx, id) for (idx, id) in enumerate(imgIds))
    data = yaml.load(open("./bluebirds/labels.yaml"))
    dinfo = {'numImg': numImg, 'numTrial': numTrial}
    dinfo['gt'] = [gtLabels[id] for id in imgIds]
    wkrIds = data.keys();
    wkrId2Idx = dict((idx, id) for (idx, id) in enumerate(wkrIds))
    print("[ data ] Dataset has ", len(wkrIds), " woker nodes, ", len(imgIds), "task nodes.")
    print("[ data ] Now building the original graph......")
    Graph = np.zeros((len(wkrIds), len(imgIds)))
    for i in range(len(wkrIds)):
        for j in range(len(imgIds)):
            Graph[i][j] = int(data[wkrId2Idx[i]][imgId2Idx[j]])
    print("[ data ] Build Graph Finished. ")
    print("[ data ] Now get the true labels...")
    TrueLabels = np.zeros(len(imgIds))
    for i in range(len(imgIds)):
        TrueLabels[i] = gtLabels[imgId2Idx[i]]
    return Graph2Edgelist(Graph), 147, 2, 39, 108, TrueLabels, "bluebird", Graph2r(Graph, len(imgIds), 2, len(wkrIds))


def read_duck():
    print("[ data ] Now reading the Duck dataset...")
    f = open("./ducks/labels.txt")
    ducks_info = f.readlines()

    f2 = open("./ducks/map.yaml")
    gtLabels = yaml.load(f2)
    workerId = gtLabels['wkr']
    wkr_num = len(workerId.keys())
    workerId2Idx = dict((id, idx) for (idx, id) in enumerate(workerId.values()))
    print(workerId2Idx)
    graph = np.zeros(wkr_num * 240)
    graph.shape = [wkr_num, 240]
    graph -= 1
    # count_wkr=np.zeros(60)
    print("[ data ] Building the Graph......")
    for i in range(len(ducks_info)):
        x = ducks_info[i]
        x = str(x).split("\n")[0]  # .split[" "]]
        x = [int(j) for j in str(x).split(" ")]
        graph[workerId2Idx[x[1]] - 1][x[0] - 1] = x[2]

    print("[ data ] Graph built finished.")
    f3 = open("./ducks/classes.yaml")
    Labels = yaml.load(f3)
    true_labels = np.zeros(240)
    for i in range(240):
        true_labels[i] = (int(Labels['labels'][i]) != 3)

    print("[ data ]  Now getting the true labels.")

    return Graph2Edgelist(graph), 280, 2, 40, 240, true_labels, "duck", Graph2r(graph, 240, 2, 40)


def read_Websites():
    print("[ data ] Now loading the Website dataset......")
    filename = "./web_processed_data_feature_2"
    # filename = "bluebird_data"
    data_all = np.load(filename + '.npz')
    user_labels = data_all['user_labels']
    true_labels = data_all['true_labels']

    source_num = data_all['source_num']
    n_samples, _ = np.shape(true_labels)
    Graph = np.zeros(source_num * n_samples) - 1
    Graph.shape = [source_num, n_samples]
    true_labels.shape = n_samples
    print("[ data] Now building the Graph.......")
    for i in range(source_num):
        for j in range(n_samples):
            for k in range(5):
                if (user_labels[j][5 * i + k] == 1):
                    Graph[i][j] = k
    print(Graph.shape)
    print("[ data ] Building finished.")
    return Graph2Edgelist(Graph), n_samples + source_num, 5, source_num, n_samples, true_labels, "web", Graph2r(Graph,
                                                                                                                n_samples,
                                                                                                                5,
                                                                                                                source_num)

def read_BlueBirds():
    # 构建整幅图，39*104
    # 返回图的邻接矩阵
    print("[ data ] Now loading blueBirds dataset......")
    numTrial = 40
    f = open("./bluebirds/gt.yaml")
    gtLabels = yaml.load(f)
    imgIds = gtLabels.keys();
    numImg = len(gtLabels)
    imgId2Idx = dict((idx, id) for (idx, id) in enumerate(imgIds))
    data = yaml.load(open("./bluebirds/labels.yaml"))
    dinfo = {'numImg': numImg, 'numTrial': numTrial}
    dinfo['gt'] = [gtLabels[id] for id in imgIds]
    wkrIds = data.keys();
    wkrId2Idx = dict((idx, id) for (idx, id) in enumerate(wkrIds))
    print("[ data ] Dataset has ", len(wkrIds), " woker nodes, ", len(imgIds), "task nodes.")
    print("[ data ] Now building the original graph......")
    Graph = np.zeros((len(wkrIds), len(imgIds)))
    for i in range(len(wkrIds)):
        for j in range(len(imgIds)):
            Graph[i][j] = int(data[wkrId2Idx[i]][imgId2Idx[j]])
    print("[ data ] Build Graph Finished. ")
    print("[ data ] Now get the true labels...")
    TrueLabels = np.zeros(len(imgIds))
    for i in range(len(imgIds)):
        TrueLabels[i] = gtLabels[imgId2Idx[i]]
    return Graph2Edgelist(Graph), 147, 2, 39, 108, TrueLabels, "bluebird", Graph2r(Graph, len(imgIds), 2, len(wkrIds))


def read_Flowers():
    print("[ data ] Now loading the Flower dataset......")
    filename = "./flower_data"
    data_all = np.load(filename + '.npz')
    user_labels = data_all['user_labels']
    true_labels = data_all['true_labels']
    source_num = data_all['source_num']
    n_samples, _ = np.shape(true_labels)

    Graph = np.zeros(source_num * n_samples) - 1
    Graph.shape = [source_num, n_samples]
    true_labels.shape = n_samples
    print("[ data] Now building the Graph.......")
    for i in range(source_num):
        for j in range(n_samples):
            for k in range(2):
                if (user_labels[j][2 * i + k] == 1):
                    Graph[i][j] = k
    print("[ data ] Building finished.")
    return Graph2Edgelist(Graph), source_num + n_samples, 2, source_num, n_samples,true_labels, "flower", Graph2r(Graph,
                                                                                                        n_samples, 2,
                                                                                                        source_num)


def build_graph_from_triplets(num_nodes, num_rels, triplets):
    """ Create a DGL graph. The graph is bidirectional because RGCN authors
        use reversed relations.
        This function also generates edge type and normalization factor
        (reciprocal of node incoming degree)
    """
    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    src, rel, dst = triplets
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel = np.concatenate((rel, rel + num_rels))
    edges = sorted(zip(dst, src, rel))
    dst, src, rel = np.array(edges).transpose()
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    print("# nodes: {}, # edges: {}".format(num_nodes, len(src)))
    return g, rel, norm


def comp_deg_norm(g):
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    return norm
