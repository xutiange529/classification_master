#!/usr/bin/python
# -*- coding: utf-8 -*-

# from __future__ import print_function

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICE'] = '0'

import time
import shutil
from datetime import timedelta
import tensorflow as tf
from cnn_model import TCNNConfig, TextCNN
from rnn_model import TRNNConfig, TextRNN
from data.data_loader import preprocess_data, read_vocab, read_category, batch_iter, process_file, build_vocab
from data.SavedModelBuilder import SavedModelBuilder

preprocess_data()
base_dir = 'data'
data_dir = os.path.join(base_dir, 'data.xlsx')
train_dir = os.path.join(base_dir, 'train.xlsx')
test_dir = os.path.join(base_dir, 'test.xlsx')
val_dir = os.path.join(base_dir, 'val.xlsx')
label_dir = os.path.join(base_dir, 'label.txt')
vocab_dir = os.path.join(base_dir, 'vocab.txt')
t = str(int(time.time()))
save_dir = 'model_path'
save_path = os.path.join(save_dir, t)  # 最佳验证结果保存路径


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, x_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.000
    total_acc = 0.000
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len
    return total_loss / data_len, total_acc / data_len


def train():
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = 'tensorboard'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    print("Loading training and validation data...")
    # 载入训练集与验证集
    start_time = time.time()
    x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, config.seq_length)
    x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, config.seq_length)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    builder = SavedModelBuilder(save_path)
    writer.add_graph(session.graph)  # have a try

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 2000  # 如果超过2000轮未提升，提前结束训练

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)

            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, x_val, y_val)  # todo

                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch

                    if os.path.exists(save_path):
                        shutil.rmtree(save_path)
                        os.makedirs(save_path)
                    builder.add_meta_graph_and_variables(session, [tf.saved_model.tag_constants.TRAINING])
                    builder.save()
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            feed_dict[model.keep_prob] = config.dropout_keep_prob
            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:
            break


if __name__ == '__main__':
    input = input('please input CNN or RNN')

    if input == 'CNN':
        print('train CNN model...')
        config = TCNNConfig()
        if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
            build_vocab(data_dir, vocab_dir, config.vocab_size)
        categories, cat_to_id = read_category(label_dir)
        words, word_to_id = read_vocab(vocab_dir)
        config.vocab_size = len(words)
        model = TextCNN(config)
        train()

    elif input == 'RNN':
        print('train RNN model...')
        config = TRNNConfig()
        if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
            build_vocab(train_dir, vocab_dir, config.vocab_size)
        categories, cat_to_id = read_category(label_dir)
        words, word_to_id = read_vocab(vocab_dir)
        config.vocab_size = len(words)
        model = TextRNN(config)
        train()

    else:
        print('input wrong,please input CNN or RNN')
