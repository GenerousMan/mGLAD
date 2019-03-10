import numpy as np

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
    return Graph.shape,source_num,n_samples,5,Graph,true_labels

read_Flowers()