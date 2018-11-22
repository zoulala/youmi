
"""
cnn1d   train val:86.7%
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os


class Config(object):
    """CNN配置参数"""
    """CNN配置参数"""
    file_name = 'cnn1'  #保存模型文件

    embedding_dim = 64  # 词向量维度
    seq_length = 30  # 序列长度
    num_classes = 2  # 类别数
    num_filters = 256  # 卷积核数目
    kernel_size = 3  # 卷积核尺寸
    vocab_max_size = 10000  # 词汇表达小

    hidden_dim = 128  # 全连接层神经元

    train_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 64  # 每批训练大小
    max_steps = 20000  # 总迭代batch数

    log_every_n = 20  # 每多少轮输出一次结果
    save_every_n = 100  # 每多少轮校验模型并保存


class Model(object):
    def __init__(self, config, vocab_size):
        self.config = config
        self.vocab_size = vocab_size

        # 待输入的数据
        self.query_seqs = tf.placeholder(tf.int32, [None, self.config.seq_length], name='query')
        # self.query_length = tf.placeholder(tf.int32, [None], name='query_length')

        self.response_seqs = tf.placeholder(tf.int32, [None, self.config.seq_length], name='response')
        # self.response_length = tf.placeholder(tf.int32, [None], name='response_length')

        self.targets = tf.placeholder(tf.float32, shape=[None], name='targets')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # 两个全局变量
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.global_loss = tf.Variable(0, dtype=tf.float32, trainable=False, name="global_loss")

        # Ann模型
        self.cnn()

        # 初始化session
        self.saver = tf.train.Saver()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())


    def cnn(self):
        """CNN模型"""
        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_max_size, self.config.embedding_dim])
            embedding_zero = tf.constant(0, dtype=tf.float32, shape=[1, self.config.embedding_dim])
            embedding = tf.concat([embedding, embedding_zero], axis=0)  # 增加一行0向量，代表padding向量值
            embedding_query = tf.nn.embedding_lookup(embedding, self.query_seqs)
            embedding_response = tf.nn.embedding_lookup(embedding, self.response_seqs)
            self.embs = embedding


        with tf.name_scope("conv"):
            # CNN layer
            cv_q = tf.layers.conv1d(embedding_query, self.config.num_filters, self.config.kernel_size, name='conv_query',padding='SAME')
            cv_r = tf.layers.conv1d(embedding_response, self.config.num_filters, self.config.kernel_size, name='conv_response',padding='SAME')

            # # stride max pooling
            # convs = tf.expand_dims(cv_q, axis=-1)  # shape=[?,596,256,1]
            # smp = tf.nn.max_pool(value=convs, ksize=[1, 3, num_filters, 1], strides=[1, 2, 1, 1],
            #                      padding='SAME')  # shape=[?,299,256,1]
            # smp = tf.squeeze(smp, -1)  # shape=[?,299,256]
            # h = tf.shape(smp)[1]
            # smp = tf.reshape(smp, shape=(-1, h * 256))

            # global max pooling layer
            gmp_q = tf.reduce_max(cv_q, reduction_indices=[1], name='gmp_q')
            gmp_r = tf.reduce_max(cv_r, reduction_indices=[1], name='gmp_r')


        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc_q = tf.layers.dense(gmp_q, self.config.hidden_dim, name='fc_q')
            fc_q = tf.contrib.layers.dropout(fc_q, self.keep_prob)
            self.fc_q = tf.nn.relu(fc_q)

            fc_r = tf.layers.dense(gmp_r, self.config.hidden_dim, name='fc_r')
            fc_r = tf.contrib.layers.dropout(fc_r, self.keep_prob)
            self.fc_r = tf.nn.relu(fc_r)

            fc = tf.concat([self.fc_q, self.fc_r], axis=-1,name='fc')
            # 分类器
            fc2 = tf.layers.dense(fc, 1, name='fc2')
            self.logits = tf.squeeze(fc2)

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.targets)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss, global_step=self.global_step)

        with tf.name_scope("score"):
            self.y_cos = tf.sigmoid(self.logits)
            self.y_pre = tf.round(self.y_cos)
            # self.y_cos = self.logits[:, -1]
            # self.y_pre = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

    def load(self, checkpoint):
        self.saver.restore(self.session, checkpoint)
        print('Restored from: {}'.format(checkpoint))

    def train(self, batch_train_g, model_path, val_g):
        with self.session as sess:
            for q, q_len, r, r_len, y in batch_train_g:
                start = time.time()
                feed = {self.query_seqs: q,
                        self.response_seqs: r,
                        self.targets: y,
                        self.keep_prob: self.config.train_keep_prob}
                batch_loss, _, y_cos ,y_pre,logits= sess.run([self.loss, self.optim, self.y_cos, self.y_pre,self.logits],
                                                       feed_dict=feed)

                end = time.time()

                # control the print lines
                if self.global_step.eval() % self.config.log_every_n == 0:
                    print('step: {}/{}... '.format(self.global_step.eval(), self.config.max_steps),
                          'loss: {:.4f}... '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end - start)))

                if (self.global_step.eval() % self.config.save_every_n == 0):
                    y_pres = np.array([])
                    y_coss = np.array([])
                    y_s = np.array([])
                    for q, q_len, r, r_len, y in val_g:
                        feed = {self.query_seqs: q,
                                self.response_seqs: r,
                                self.targets: y,
                                self.keep_prob: 1}

                        y_pre, y_cos = sess.run([self.y_pre, self.y_cos], feed_dict=feed)
                        y_pres = np.append(y_pres, y_pre)
                        y_coss = np.append(y_coss, y_cos)
                        y_s = np.append(y_s, y)

                    # 计算预测准确率
                    print('val len:', len(y_s))
                    print("accuracy:{:.2f}%.".format((y_s == y_pres).mean() * 100),
                          'best:{:.2f}%'.format(self.global_loss.eval()* 100))

                    acc_val = (y_s == y_pres).mean()
                    if acc_val > self.global_loss.eval():
                        print('save best model...')
                        update = tf.assign(self.global_loss, acc_val)  # 更新最优值
                        sess.run(update)
                    self.saver.save(sess, os.path.join(model_path, 'model'), global_step=self.global_step)

                if self.global_step.eval() >= self.config.max_steps:
                    break


    def test_one_by_one(self, a_sample):
        '''
        input:q,r
        :return:0 or 1 
        '''
        sess = self.session
        q, q_len, r, r_len, y = a_sample
        feed = {self.query_seqs: np.reshape(q,[1,-1]),
                self.response_seqs: np.reshape(r,[1,-1]),
                self.keep_prob: 1}
        y_pre, y_cos = sess.run([self.y_pre, self.y_cos], feed_dict=feed)
        return y_pre[0],y_cos[0]


    def get_one_state(self, a_q):
        '''
        input:q
        :return:q_state 
        '''
        sess = self.session
        q, q_len = a_q
        feed = {self.query_seqs: np.reshape(q,[1,-1]),
                self.keep_prob: 1}
        q_state = sess.run(self.fc_q, feed_dict=feed)
        return q_state


    def get_mul_state(self, libs_arrs):
        sess = self.session
        response_matul_state = np.empty([1, self.config.hidden_dim ])
        n = len(libs_arrs)
        for i in range(n):
            feed = {self.response_seqs: libs_arrs[i][0].reshape(-1, self.config.seq_length),
                    self.keep_prob: 1.}
            response_one_state = sess.run(self.fc_r, feed_dict=feed)
            response_matul_state = np.append(response_matul_state, response_one_state, axis=0)
            if i % 1000 == 0:
                print(i)
        response_matul_state = np.delete(response_matul_state, 0, 0)
        print('libs caculate ok')
        return response_matul_state

    def test(self, q_state, response_matul_state):
        sess = self.session
        y_coses = []
        for r_state in response_matul_state:
            feed = {self.fc_q: q_state.reshape(-1, self.config.hidden_dim),
                    self.fc_r: r_state.reshape(-1, self.config.hidden_dim),
                    self.keep_prob: 1.}
            cos = sess.run(self.y_cos, feed_dict=feed)
            y_coses.append(cos)
        return y_coses

    def save_embeds(self):
        with self.session as sess:
            embs= sess.run(self.embs)
        return embs



