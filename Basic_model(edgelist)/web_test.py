def read_Websites():
    filename = "web_processed_data_feature_2"
    # filename = "age_data_3_category"
    # filename = "bluebird_data"
    data_all = np.load(filename + '.npz')
    user_labels = data_all['user_labels']
    print(user_labels)
    
    label_mask = data_all['label_mask']
    true_labels = data_all['true_labels']
    category_size = data_all['category_num']
    source_num = data_all['source_num']
    n_samples, _ = np.shape(true_labels)
    #return Graph.shape,len(wkrIds),len(imgIds),2,Graph,TrueLabels
    return 

read_Websites()