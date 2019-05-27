from scipy.sparse import issparse
import numpy as np
from numpy.matlib import repmat
import torch

def get_majority_y(user_labels, source_num, category_num):
    if not issparse(user_labels):
        #         n_samples, source_mul_category = np.shape(user_labels)
        #         source_num = source_mul_category / category_num
        tmp = np.eye(category_num)
        template = repmat(tmp, source_num, 1)
        majority_y = np.matmul(user_labels, template)
        majority_y = np.divide(majority_y, np.matlib.repmat(np.sum(majority_y, 1, keepdims=True), 1, category_num))
    else:
        user_labels = user_labels.todense()
        tmp = np.eye(category_num)
        template = repmat(tmp, source_num, 1)
        majority_y = np.matmul(user_labels, template)
        majority_y = np.divide(majority_y, np.matlib.repmat(np.sum(majority_y, 1), 1, category_num))
    return majority_y


def get_constant_y(batch_size, category_size):
    constant_y = {}
    for i in range(category_size):
        constant_tmp = torch.zeros((batch_size, category_size))
        constant_tmp[:, i] = 1.0
        # constant_y[i] = tf.constant(constant_tmp)  # 这个地方不清楚是否要用torch.nn.nn.constant改
        constant_y[i] = constant_tmp
    return constant_y
    # 返回一个dict 里面有K个M*K的矩阵，第i个矩阵的第i列都是1，其他为0
