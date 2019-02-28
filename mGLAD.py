# Placeholders中包含:
# 1.edge信息，形状为K，记录了每个点之间的连接关系，是一个输入的值
# 2.edge类型，形状为1，其值设为x，代表有x个不同label的可能
# 3.label信息，形状待定
import numpy as np
import tensorflow as tf
import yaml

def Cal_ProbLoss(loss,P,edges):
    #predict_edges' shape:(K*x)
    
    def cond_worker(i,loss_now):
        #判断第i个worker

        return i < edges.shape[0]
    def body_worker(i,loss_now):
        #对loss进行累加运算
        def cond_task(j,loss_now):
            #判断第j个task

            return j<edges.shape[1]
        def body_task(j,loss_now):
            #对loss进行累加运算
            loss=tf.add(loss,P[i][j][edges[i][j]])

            return j+1, loss 
        loss=tf.while_loop(cond_task,body_task,[0,loss])

        return i+1, loss 

    loss=tf.while_loop(cond_worker,body_worker,[0,loss])

    return loss

def read_BlueBirds():
    #构建整幅图，39*104
    #返回图的邻接矩阵

    print("[ data ] Now loading blueBirds dataset......")
    numTrial=40
    f=open("C://Users//Administrator//Desktop//demo//bluebirds//gt.yaml")
    gtLabels = yaml.load(f)
    imgIds = gtLabels.keys(); numImg = len(gtLabels)
    imgId2Idx = dict((idx, id) for (idx, id) in enumerate(imgIds))
    data = yaml.load(open("C://Users//Administrator//Desktop//demo//bluebirds//labels.yaml"))
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
    return Graph

class mGLAD(Model):
    # inputs: placeholders['edges'] 形状为K
    # outputs: 形状为task节点数*总类数x

    def __init__(self, placeholders, input_dim, **kwargs):
        super(mGLAD, self).__init__(**kwargs)

        self.inputs = placeholders['edges']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].Vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error

        #此处计算输出边的连接关系和原本边的关系的交叉熵
        #但是原文意思是要计算概率，也就是每条边和原本相同的值的概率
        #TO DO：确定outputs的输出形状，如果原始边连接矩阵的形状是K，总共有x个label，那就应该是K*x，然后softmax找原始Label概率

        self.loss += Cal_ProbLoss(self.loss, self.outputs, self.placeholders['edges'])

    def _accuracy(self):
        self.accuracy =  tf.reduce_mean(tf.equal(tf.argmax(self.outputs, 2), self.placeholders['edges']))

    def _build(self):

        print("[ model ] Building mGLAD model......")
        print("[ model ] Appending MPNN layer......")
        self.layers.append(mpnn(input_dim = self.input_dim,
                                output_dim = FLAGS.hidden1,
                                placeholders = self.placeholders,
                                update_step = 5,
                                Logging = self.logging))
        print("[ model ] Appending Decoder layer......")
        self.layers.append(Decoder(input_dim = FLAGS.hidden1,
                                    output_dim = self.output_dim,
                                    placeholders = self.placeholders))
        print("[ model ] Build finished.")

    def predict(self):
        return tf.nn.softmax(self.outputs)

class mpnn(Layer):
    # input_dim=(39*104)
    # output_dim=[(39*D),(104*L)]

    def __init__(self, input_dim, output_dim, placeholders, update_step, **kwargs):
        super(mpnn, self).__init__(**kwargs)
        #self.update_function_a = update_a
        #self.update_function_t = update_t

        self.adj = placeholders['edges']
        self.step= update_step
        self.input_dim=input_dim
        ## 定义基本变量
        ## adj: 邻接矩阵
        ## step: 更新次数
        ## output: 输出 

        with tf.variable_scope('mpnn_vars'):
            #这个地方要加入可训练参数，在mpnn层里的

            #TO DO: 确定形状
            #此处形状待定，暂时先按下面的来。

            self.Vars["Awij"]=tf.Variable(initial_value=tf.truncated_normal(shape=[placeholders['edge_type'],placeholders['ability_num'],placeholders['edge_type']], mean=0, stddev=1), name="Awij")
            #在这里的形状是（ability类别*edge类别），即10*2

            self.Vars["Awij2"]=tf.Variable(initial_value=tf.truncated_normal(shape=[placeholders['edge_type'], 1, placeholders['ability_num']], mean=0, stddev=1), name="Awij2")
            #在这里的形状是（1*每个人的ability类别），此处是1*10

        #if self.logging:
        #    self._log_vars()

    def _call(self, inputs):
        # 进行调用，会把输入图给导进来，然后调用下面的更新函数
        # TODO : 
        # 1.务必确定初始的a,t形状
        # 2.务必确定更新函数的写法，有些复杂。
        
        # 难点：
        # **更新a：是否要把不同label的累加矩阵算上
        # **更新t：确定只加正确类的label的权重

        first_a = tf.random_normal(shape=[self.input_dim[0], self.placeholders['ability_num']], stddev=1, seed=1)
        first_t = tf.random_normal(shape=[self.input_dim[1], self.placeholders['edge_type']], stddev=1, seed=1)

        def cond(i,update_a, update_t):
            return i<self.step

        def body(i,update_a,update_t):
            def cond_a(j,update_a,update_t):
                # 判断a的更新进行完毕与否

                return j < self.input_dim[0]

            def cond_t(k,update_a,update_t):
                # 判断t的更新进行完毕与否

                return k < self.input_dim[1]

            def body_a(j,update_a,update_t):
                # 用于循环迭代每一轮a的更新
                def cond_tau(jj,update_a,update_t):
                    #用于判断是否迭代完了每一个worker

                    return jj < self.input_dim[1]

                def body_tau(jj,update_a,update_t):
                    # 用于针对每一个worker的ability，都根据原始label选择对应矩阵相乘
                    now_t=update_t[jj][inputs[j][jj]] # 根据这条边是啥取用指定tau里的值，维度 1*1
                    
                    A2_label=tf.cond(inputs[j][jj]==0,self.Vars["Awij2"][0],self.Vars["Awij2"][1])
                    # 根据不同label 选用不同的A2

                    update_a[j]=tf.add(update_a[j],tf.multiply(now_t, A2_label))
                    #将update_a的第j个worker的得分进行一个累加计算

                    return jj+1, update_a, update_t

                _, update_a, update_t = tf.while_loop(cond_tau, body_tau, [0,update_a, update_t])

                return j+1, update_a, update_t 

            def body_t(k,update_a,update_t):
                
                # 用于循环迭代每一轮t的更新,迭代task个数次，是第二个维度

                def cond_aj(kk,update_a,update_t):
                    # 用于判断是否考虑到了每一个worker
                    return kk < self.input_dim[0]

                def body_aj(kk,update_a,update_t):
                    A_label=tf.cond(inputs[kk][k]==0,self.Vars["Awij"][0],self.Vars["Awij"][1])
                    # 根据不同label取用不同矩阵，存在本次A_label中

                    update_t[k]=tf.add(update_t[k], # 求和的过程，从kk=0到kk=最后一个worker的序号，都给加上
                        tf.multiply(update_a[kk],A_label))
                    # 1*10 x 10*2  = 1*2
                    return kk + 1, update_a, update_t
                _, update_a, update_t = tf.while_loop(cond_aj, body_aj, [0,update_a, update_t])

                return k + 1, update_a, update_t

            # TO DO:这个地方维度计算没有敲定，还有转置等操作要做
            _, update_a, update_t = tf.while_loop(cond_a, body_a, [0,update_a, update_t])
            # 对t进行一轮更新

            _, update_a, update_t = tf.while_loop(cond_t, body_t, [0,update_a, update_t])
            # 对a进行一轮更新

            return i+1, update_a, update_t

        i, final_a, final_t=tf.while_loop(cond, body, [0, first_a, first_t])

        padding_t=tf.constant(0,shape=[self.input_dim[0],self.placeholders["ability_num"]])
        padding_a=tf.constant(0,shape=[self.input_dim[1],self.placeholders["edge_type"]])

        #此处为了保证输出是一个完整的张量,于是分别对a,t进行padding操作，让它们的维度均为：[-1, worker特征数 + task特征数]
        output=tf.concat(tf.concat(final_a,padding_a,axis=1), tf.concat(final_t,padding_t,axis=1), axis=0 )
        # 拼接后即可传出output
        # 维度为：（ worker node+ task node , worker feature, task feature ）

        return output

class Decoder(Layer):
    def __init__(self, input_dim, output_dim, placeholders, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.placeholders = placeholders
        self.P = tf.random_normal(shape=[self.input_dim[0], self.input_dim[1], 2], stddev=1, seed=1)
        
        with tf.variable_scope('Decoder_vars'):
            self.Vars["W"]=tf.Variable(initial_value=tf.truncated_normal(shape=[self.placeholders['ability_num'], 1], mean=0, stddev=1), name="W")
            #在这里的形状是（ability类别*1），即10*1
             self.Vars["b"]=tf.Variable(initial_value=tf.truncated_normal(shape=1, mean=0, stddev=1), name="b")
            #在这里的形状是 1

    def _call(self, inputs):
            worker_feature = inputs[0:self.placeholders["worker_num"]][0:self.placeholders["ability_num"]]
            task_feature   = inputs[self.placeholders["worker_num"]:][0:self.placeholders["edge_type"]] 
            
            #进入循环，此处循环较为复杂，要进行每个worker 对每个task 的每个label的可能性计算

            def cond_worker(i, P):
                # 针对第i个worker

                return i < self.placeholders["worker_num"]
            
            def body_worker(i, P):
                def cond_task(j, P):
                    # 针对第j个task
                    
                    return j < self.placeholders["task_num"]
                
                def body_task(j,P):
                    
                    def cond_label(l,P):
                        # 针对第l个label
                        return l < self.placeholders["edge_type"]
                    
                    def body_label(l,P):
                        # 对每个l进行计算
                        tau_prob = task_feature[j][l]
                        prob_part1 = tf.sigmoid(
                                            tf.add(
                                                tf.multiply(worker_feature[i],self.Vars["W"]),
                                                self.Vars["b"])
                                        )
                        prob_part2 = tf.divide(
                                        tf.subtract(tf.constant(1),prob_part1),
                                        tf.subtract(self.placeholders["edge_type"],tf.constant(1))
                                        )

                        P[i][j][l]=tf.mul(
                                        tf.pow(prob_part1,tau_prob),
                                        tf.pow(
                                            prob_part2,
                                            tf.subtract(
                                                tf.constant(1),
                                                tau_prob)
                                            )
                                        )
                        return l+1, P
                    _, P = tf.while_loop(cond_label,body_label,[0,P])

                    return j+1, P
                _,P = tf.while_loop(cond_task,body_task,[0,P])

                return i+1, P
            _, P= tf.while_loop(cond_worker,body_worker,[0,self.P])

            return P
            #进行调用

if __name__ == '__main__':
    BB_Graph=read_BlueBirds()

    #placeholders=construct_placeholders(np.randint(2,size=(10,10)))
    #mpnn_test=mpnn([10,10],[10,10,20],placeholders,5)
