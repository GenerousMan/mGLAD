import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import utils

class laa_model(nn.Module):
    def __init__(self, source_num, n_samples, input_size, category_size, batch_size, y_kl_strength):
        super(laa_model, self).__init__()
        self.source_num = source_num
        self.n_samples = n_samples
        self.input_size = input_size
        self.category_size = category_size
        self.batch_size = batch_size
        self.y_kl_strength = y_kl_strength

        # parameters for decoder
        self.weights_reconstr = nn.Parameter(torch.Tensor(category_size, input_size))
        self.biases_reconstr = nn.Parameter(torch.Tensor(1, input_size))
        nn.init.xavier_uniform_(self.weights_reconstr, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.biases_reconstr, gain=nn.init.calculate_gain('relu'))

        # parameters for encoder i.e. classifier
        self.weights_y_classifier = nn.Parameter(torch.Tensor(input_size, category_size))
        self.biases_y_classifier = nn.Parameter(torch.Tensor(1, category_size))
        nn.init.xavier_uniform_(self.weights_y_classifier, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.biases_y_classifier, gain=nn.init.calculate_gain('relu'))

        # TODO: ability
        self.ability = nn.Parameter(torch.Tensor(1, input_size))
        nn.init.xavier_uniform_(self.ability, gain=nn.init.calculate_gain('relu'))

        # TODO: object ambiguity


        # TODO: Latent Aspect


    def decoder(self, x, mask):
        # define source0-wise template      TODO: 用于做source-wise softmax
        source_wise_template = torch.zeros((self.input_size, self.input_size)).float()
        for i in range(self.input_size):
            # [i*K:(i+1)*K, i*K:(i+1)*K]
            source_wise_template[i * self.category_size:(i + 1) * self.category_size,
            i * self.category_size:(i + 1) * self.category_size] = 1

        def reconstruct_x_y(y):
            x_reconstr_tmp = torch.add(torch.matmul(y, self.weights_reconstr), self.biases_reconstr)  # 公式（3）
            x_reconstr = torch.div(torch.exp(x_reconstr_tmp),
                                   torch.matmul(torch.exp(x_reconstr_tmp),
                                                source_wise_template))  # source-wise softmax operator  具体怎么做的还没想明白，先放一边
            return x_reconstr

        # define constant_y  # TODO 以下部分不太明白
        constant_y = utils.get_constant_y(self.batch_size, self.category_size)  # M * K   TODO：decoder阶段采样?

        # constant_y[0]:
        # [1., 0.]
        # [1., 0.]
        # ...
        #
        # constant_y[1]:
        # [0., 1.]
        # [0., 1.]
        # ...

        tmp_reconstr = []
        for i in range(self.category_size):
            _tmp_reconstr_x = reconstruct_x_y(constant_y[i])
            _tmp_cross_entropy = - torch.mul(x, torch.log(1e-10 + _tmp_reconstr_x))               # 公式(4)
            tmp_reconstr.append(
                torch.mean(torch.mul(mask, _tmp_cross_entropy), 1).unsqueeze(1))
        reconstr_x = torch.cat(tuple(tmp_reconstr), 1)  # 108,2
        return reconstr_x

    def encoder(self, x):
        # print("............................",(x.matmul(self.weights_y_classifier).unsqueeze(1).shape))
        y_classifier = F.softmax((x.matmul(self.weights_y_classifier)) + self.biases_y_classifier)  # TODO: original
        # y_classifier = F.softmax(self.ability.mul(x.unsqueeze(1)).squeeze(1).matmul(self.weights_y_classifier) + self.biases_y_classifier)  # TODO: 加上ability
        return y_classifier

    def get_loss_x_y(self, x, y_target):
        # loss x->y
        y_classifier = self.encoder(x)
        _tmp_classifier_cross_entropy = - torch.mul(y_target, torch.log(1e-10 + y_classifier))
        loss_classifier_x_y = torch.mean(torch.sum(_tmp_classifier_cross_entropy, 1))
        return y_classifier, loss_classifier_x_y

    def get_loss(self, x, mask, y_target):
        y_classifier = self.encoder(x)
        reconstr_x = self.decoder(x, mask)

        # classifier            weight loss
        loss_w_classifier_l1 = torch.sum(torch.abs(self.weights_y_classifier))  # TODO： 不全

        _tmp_loss_backprop = torch.mul(y_classifier, reconstr_x)
        loss_classifier_y_x = torch.mean(torch.sum(_tmp_loss_backprop, 1))
        y_prior = y_target
        loss_y_kl = torch.mean(torch.sum(
            torch.mul(y_classifier, torch.log(1e-10 + y_classifier)) - torch.mul(y_classifier,
                                                                                 torch.log(1e-10 + y_prior)), 1))
        y_kl_strength = self.y_kl_strength
        # use proper parameters
        loss_classifier = loss_classifier_y_x + 0.0001 * loss_y_kl + 0.005 / self.source_num / self.category_size / self.category_size * loss_w_classifier_l1

        return y_classifier, loss_classifier
