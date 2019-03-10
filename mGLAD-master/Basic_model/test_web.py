import numpy as np

def read_Websites():
    filename = "./web_processed_data_feature_2"
    # filename = "age_data_3_category"
    # filename = "bluebird_data"
    data_all = np.load(filename + '.npz')
    user_labels = data_all['user_labels']
    print(user_labels[:,5:10])
    label_mask = data_all['label_mask']
    print(label_mask)
    true_labels = data_all['true_labels']

    category_size = data_all['category_num']
    source_num = data_all['source_num']
    n_samples, _ = np.shape(true_labels)
    #return Graph.shape,len(wkrIds),len(imgIds),2,Graph,TrueLabels
    Graph=np.zeros(source_num*n_samples)-1
    Graph.shape=[source_num,n_samples]
    true_labels.shape=n_samples
    for i in range(source_num):
        for j in range(n_samples):
            for k in range(5):
                if(user_labels[j][5*i+k]==1):
                    Graph[i][j]=k
    print(Graph)
    return Graph.shape,source_num,n_samples,5,Graph,true_labels
