# coding: utf-8
from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os

class Config(object):
    """RNN配置参数"""
    file_name = 'dense'  #保存模型文件

    embedding_dim = 200      # 词向量维度
    seq_length = 16        # 序列最大长度
    # num_classes = 2        # 类别数
    vocab_max_size = 10000       # 词汇表达小

    num_layers= 1           # 隐藏层层数
    hidden_dim = 200        # 隐藏层神经元
    # rnn = 'gru'             # lstm 或 gru

    train_keep_prob = 0.75  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 100  # 每批训练大小
    max_steps = 20000  # 总迭代batch数

    batch_norm = True  # fc层是否进行batch归一化

    log_every_n = 20  # 每多少轮输出一次结果
    save_every_n = 100  # 每多少轮校验模型并保存


class Model(object):

    def __init__(self, config, vocab_size):
        self.config = config
        self.vocab_size = vocab_size

        # 待输入的数据
        self.query_seqs = tf.placeholder(tf.int32, [None, self.config.seq_length], name='query')
        self.query_length = tf.placeholder(tf.int32, [None], name='query_length')

        self.response_seqs = tf.placeholder(tf.int32, [None, self.config.seq_length], name='response')
        self.response_length = tf.placeholder(tf.int32, [None], name='response_length')

        self.targets = tf.placeholder(tf.int32, shape=[None], name='targets')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # 两个全局变量
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.global_loss = tf.Variable(0, dtype=tf.float32, trainable=False, name="global_loss")

        # Ann模型
        self.rnn()

        # 初始化session
        self.saver = tf.train.Saver()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def rnn(self):
        """rnn模型"""

        # 词嵌入层
        # self.lstm_query_seqs = tf.one_hot(self.query_seqs, depth=self.vocab_size)  # 独热编码[1,2,3] depth=5 --> [[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0]]，此时的输入节点个数为num_classes
        # self.lstm_response_seqs = tf.one_hot(self.response_seqs, depth=self.vocab_size)
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.vocab_size, self.config.embedding_dim])
            self.embs = embedding
            # embedding_zero = tf.constant(0, dtype=tf.float32, shape=[1, self.config.embedding_dim])
            # embedding = tf.concat([embedding, embedding_zero], axis=0)  # 增加一行0向量，代表padding向量值
            self.lstm_query_seqs = tf.nn.embedding_lookup(embedding, self.query_seqs)  # 词嵌入[1,2,3] --> [[3,...,4],[0.7,...,-3],[6,...,9]],embeding[depth*embedding_size]=[[0.2,...,6],[3,...,4],[0.7,...,-3],[6,...,9],[8,...,-0.7]]，此时的输入节点个数为embedding_size
            self.lstm_response_seqs = tf.nn.embedding_lookup(embedding, self.response_seqs)


        with tf.name_scope("rnn"):

            def get_mul_cell(hidden_dim, num_layers):   # 创建多层lstm
                def get_en_cell(hidden_dim):   # 创建单个lstm
                    enc_base_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim, forget_bias=1.0)
                    return enc_base_cell
                return tf.nn.rnn_cell.MultiRNNCell([get_en_cell(hidden_dim) for _ in range(num_layers)])

            with tf.variable_scope("bilstm_q"):
                # 构建双向lstm
                query_cell_fw = get_mul_cell(self.config.hidden_dim, self.config.num_layers)
                query_cell_bw = get_mul_cell(self.config.hidden_dim, self.config.num_layers)
                query_output, self.query_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=query_cell_fw,
                                                                                 cell_bw=query_cell_bw,
                                                                                 inputs=self.lstm_query_seqs,
                                                                                 sequence_length=self.query_length,
                                                                                 dtype=tf.float32,
                                                                                 time_major=False)

            with tf.variable_scope("bilstm_r"):
                response_cell_fw = get_mul_cell(self.config.hidden_dim, self.config.num_layers)
                response_cell_bw = get_mul_cell(self.config.hidden_dim, self.config.num_layers)
                response_output, self.response_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=response_cell_fw,
                                                                                       cell_bw=response_cell_bw,
                                                                                       inputs=self.lstm_response_seqs,
                                                                                       sequence_length=self.response_length,
                                                                                       dtype=tf.float32,
                                                                                       time_major=False)

            query_c_fw, query_h_fw = self.query_state[0][-1]  # 前向最后一层输出
            query_c_bw, query_h_bw = self.query_state[1][-1]  # 后向最后一层输出
            response_c_fw, response_h_fw = self.response_state[0][-1]
            response_c_bw, response_h_bw = self.response_state[1][-1]

            self.query_h_state = tf.concat([query_h_fw, query_h_bw], axis=1)
            self.response_h_state = tf.concat([response_h_fw, response_h_bw], axis=1)



            ### INFERENCE
            ### Max pooling
            max_pool = tf.contrib.keras.layers.GlobalMaxPool1D()
            lstm1_pool = max_pool(query_output[0])  # [batch, n_steps, embed_size]-->[batch, embed_size],选取n_steps中最大的保留。如[[[1,2,3,4],[2,-1,0,6],[7,0,-2,1]],...,[[1,2,3,4],[2,-1,0,9],[0,1,-1,4]]]-->[7,2,3,6],...,[2,2,3,9]
            lstm2_pool = max_pool(response_output[0])

            ### Features
            flat1 = tf.contrib.layers.flatten(lstm1_pool)  #作用reshape：flattened = tf.reshape(x, [tf.shape(x)[0], -1])，最终维度是2:[batch,n]
            flat2 = tf.contrib.layers.flatten(lstm2_pool)
            # flat1 = self.query_h_state
            # flat2 = self.response_h_state

            mult = tf.multiply(flat1, flat2)
            diff = tf.abs(tf.subtract(flat1, flat2))

            # if self.config.feats == "raw":
            #     concat = tf.concat([flat1, flat2], axis=-1)
            # elif self.config.feats == "dist":
            #     concat = tf.concat([mult, diff], axis=-1)
            # elif self.config.feats == "all":
            #     concat = tf.concat([flat1, flat2, mult, diff], axis=-1)
            concat = tf.concat([flat1, flat2, mult, diff], axis=-1)

            ### FC layers
            self.concat_size = int(concat.get_shape()[1])
            intermediary_size = 2 + (self.concat_size - 2) // 2
            # intermediary_size = 512

            with tf.variable_scope("fc1") as scope:
                W1 = tf.Variable(tf.random_normal([self.concat_size, intermediary_size], stddev=1e-3), name="w_fc")
                b1 = tf.Variable(tf.zeros([intermediary_size]), name="b_fc")

                z1 = tf.matmul(concat, W1) + b1

                if self.config.batch_norm:
                    epsilon = 1e-3
                    batch_mean1, batch_var1 = tf.nn.moments(z1, [0])
                    scale1, beta1 = tf.Variable(tf.ones([intermediary_size])), tf.Variable(tf.zeros([intermediary_size]))
                    z1 = tf.nn.batch_normalization(z1, batch_mean1, batch_var1, beta1, scale1, epsilon)

                fc1 = tf.nn.dropout(tf.nn.tanh(z1), keep_prob=self.keep_prob)


            with tf.variable_scope("fc2") as scope:
                W2 = tf.Variable(tf.random_normal([intermediary_size, 2], stddev=1e-3), name="w_fc")
                b2 = tf.Variable(tf.zeros([2]), name="b_fc")

                z2 = tf.matmul(fc1, W2) + b2

                if self.config.batch_norm:
                    epsilon = 1e-3
                    batch_mean2, batch_var2 = tf.nn.moments(z2, [0])
                    scale2, beta2 = tf.Variable(tf.ones([2])), tf.Variable(tf.zeros([2]))
                    z2 = tf.nn.batch_normalization(z2, batch_mean2, batch_var2, beta2, scale2, epsilon)

                self.fc2 = z2

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets, logits=self.fc2)
            self.mean_loss = tf.reduce_mean(self.losses, name="mean_loss")  # batch样本的平均损失
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.mean_loss, global_step=self.global_step)

        with tf.name_scope("accuracy"):
            self.y_pre = tf.argmax(self.fc2, 1)
            self.y_cos = tf.nn.softmax(logits=self.fc2)[:, -1]

    def load(self, checkpoint):
        self.saver.restore(self.session, checkpoint)
        print('Restored from: {}'.format(checkpoint))

    def train(self, batch_train_g, model_path, val_g):
        with self.session as sess:
            for q, q_len, r, r_len, y in batch_train_g:
                start = time.time()
                feed = {self.query_seqs: q,
                        self.query_length: q_len,
                        self.response_seqs: r,
                        self.response_length: r_len,
                        self.targets: y,
                        self.keep_prob: self.config.train_keep_prob}
                batch_loss, _, y_cos ,y_pre= sess.run([self.mean_loss, self.optim, self.y_cos, self.y_pre],
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
                                self.query_length: q_len,
                                self.response_seqs: r,
                                self.response_length: r_len,
                                self.targets: y,
                                self.keep_prob: 1}

                        y_pre, y_cos = sess.run([self.y_pre, self.y_cos], feed_dict=feed)
                        y_pres = np.append(y_pres, y_pre)
                        y_coss = np.append(y_coss, y_cos)
                        y_s = np.append(y_s, y)

                    # 计算预测准确率
                    print('val len:', len(y_s))
                    print("accuracy:{:.2f}%.".format((y_s == y_pres).mean() * 100),
                          'best:{:.2f}%'.format(self.global_loss.eval()* 100) )

                    acc_val = (y_s == y_pres).mean()
                    if acc_val > self.global_loss.eval():
                        print('save best model...')
                        update = tf.assign(self.global_loss, acc_val)  # 更新最优值
                        sess.run(update)
                        self.saver.save(sess, os.path.join(model_path, 'model'), global_step=self.global_step)
                if self.global_step.eval() >= self.config.max_steps:
                    break

    def test_to_matul(self,libs_arrs):
        sess = self.session
        response_matul_state = np.empty([1,self.config.hidden_dim*2])
        n = len(libs_arrs)
        for i in range(n):
            feed = {self.response_seqs: libs_arrs[i][0].reshape(-1,self.config.seq_length),
                    self.response_length: libs_arrs[i][1].reshape(1),
                    self.keep_prob: 1.}
            response_one_state = sess.run(self.response_matul_state, feed_dict=feed)
            response_matul_state = np.append(response_matul_state, response_one_state,axis=0)
            if i%1000==0:
                print(i)
        response_matul_state = np.delete(response_matul_state,0,0)
        print('libs caculate ok')
        return response_matul_state

    def test(self,input_arr,input_len, response_matul_state):
        sess = self.session
        feed = {self.query_seqs: input_arr.reshape(-1, self.config.seq_length),
                self.query_length:input_len.reshape(1),
                self.response_matul_state: response_matul_state,
                self.keep_prob: 1.}
        logits = sess.run(self.logits, feed_dict=feed)
        n_max = np.max(logits[0])
        max_index = np.where(logits[0] == n_max)
        return max_index[0].tolist()

    def save_embeds(self):
        with self.session as sess:
            embs= sess.run(self.embs)
        return embs


