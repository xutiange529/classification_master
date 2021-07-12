# coding: utf-8

import tensorflow as tf


class TCNNConfig(object):
    """CNN配置参数"""
    embedding_dim = 128  # 词向量维度
    seq_length = 64  # 序列长度
    num_classes = 4  # 类别数
    num_filters = 128  # 卷积核数目
    filter_sizes = [3, 4, 5, 6]  # 卷积核尺寸
    vocab_size = 12000  # 词汇表达小
    hidden_dim = 256  # 全连接层神经元
    dropout_keep_prob = 0.9  # dropout保留比例
    learning_rate = 1e-4  # 学习率
    batch_size = 1024  # 每批训练大小
    num_epochs = 300  # 总迭代轮次
    print_per_batch = 1000  # 每多少轮输出一次结果
    save_per_batch = 1000  # 每多少轮存入tensorboard
    l2_reg_lambda = 0.0


class TextCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.cnn()

    def cnn(self):
        """CNN模型"""
        # 词向量映射
        embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
        embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)
        embedding_inputs = tf.expand_dims(embedding_inputs, -1)

        pooled_outputs = []
        for i, filter_size in enumerate(self.config.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.config.embedding_dim, 1, self.config.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.config.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    embedding_inputs,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.config.seq_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.config.num_filters * len(self.config.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.config.dropout_keep_prob)

        with tf.name_scope("output"):
            # 全连接层，后面接dropout以及relu激活
            l2_loss = tf.constant(0.0)
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, self.config.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.config.num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            # self.predictions = tf.argmax(tf.nn.softmax(self.scores), 1, name="predictions")

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy) + self.config.l2_reg_lambda * l2_loss
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.predictions)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
