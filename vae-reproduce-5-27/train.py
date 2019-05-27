import torch
import numpy as np
import utils
from model import laa_model

# setting
y_kl_strength = 0.0001
learning_rate = 0.005
n_epochs_x_y = 100
n_epochs = 10000

# 读数据集
filename = "bluebird_data"
# filename = "flower_data"
# filename = "web_processed_data_feature_2"

dataset_path = "dataset/"
data_all = np.load(dataset_path + filename + '.npz')
user_labels = data_all['user_labels']  # [M,(N*K)]
label_mask = data_all['label_mask']  # [M,(N*K)]
true_labels = data_all['true_labels']  # [M,1]
category_size = data_all['category_num']  # 即label种类数目 L
source_num = data_all['source_num']  # 指 worker 数目 N
n_samples, _ = np.shape(true_labels)  # n_sample 是指task的数目 M
input_size = source_num * category_size  # N * K
batch_size = n_samples  # batch_size， 一个batch就是全部的task点
total_batches = int(n_samples / batch_size)
# 这个地方是得到majority voting result
majority_y = utils.get_majority_y(user_labels, source_num, category_size)

user_labels = torch.from_numpy(user_labels).float()
label_mask = torch.from_numpy(label_mask).float()
true_labels = torch.from_numpy(true_labels.astype(float)).float()
majority_y = torch.from_numpy(majority_y).float()

# create model
model = laa_model(source_num=source_num,
                  n_samples=n_samples,
                  input_size=input_size,
                  category_size=category_size,
                  batch_size=batch_size,
                  y_kl_strength=y_kl_strength
                  )

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 画图
loss_all = []
acc_all = []
edge_all = []

epoch = 0
best_mrr = 0

print("Initialize x -> y ...")
for epoch in range(n_epochs_x_y):
    model.train()
    total_hit = 0

    for batch in range(total_batches):
        # evaluate with true labels
        batch_x, batch_mask, batch_y_label, batch_majority_y = user_labels, label_mask, true_labels, majority_y
        y_classifier, loss_classifier_x_y = model.get_loss_x_y(batch_x, batch_majority_y)
        loss_classifier_x_y.backward()
        optimizer.step()  # TODO：这里和下面必须是同一个optimizer
        optimizer.zero_grad()

        # evaluate with true labels
        inferred_category = torch.argmax(y_classifier, 1).reshape(batch_size, 1).float()
        hit_num = torch.sum((inferred_category.eq(batch_y_label)).int())

        total_hit += hit_num

    print("epoch: {0} accuracy: {1}".format(epoch, float(total_hit) / n_samples))

print("\n\n\n")

print("Train the whole network ...")
for epoch in range(n_epochs):
    total_hit = 0
    for batch in range(total_batches):
        batch_x, batch_mask, batch_y_label, batch_majority_y = user_labels, label_mask, true_labels, majority_y
        # get y_prob from classifier x -> y
        y_classifier, loss_classifier = model.get_loss(batch_x, batch_mask, batch_majority_y)
        loss_classifier.backward()
        optimizer.step()  # TODO：这里和上面必须是同一个optimizer
        optimizer.zero_grad()

        # evaluate with true labels
        inferred_category = torch.argmax(y_classifier, 1).reshape(batch_size, 1).float()
        hit_num = torch.sum((inferred_category.eq(batch_y_label)).int())

        total_hit += hit_num

    print("epoch: {0} accuracy: {1}".format(epoch, float(total_hit) / n_samples))
