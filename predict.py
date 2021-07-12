# coding: utf-8


import os
import tensorflow as tf
from cnn_model import TCNNConfig
from rnn_model import TRNNConfig
from data.data_loader import read_category, read_vocab, pad_sequences


def get_model_file(path):
    dirs = [i for i in os.listdir(path) if os.path.isdir(os.path.join(path, i))]
    _dir = sorted(dirs)[-1]
    model_path = os.path.join(path, _dir)
    return model_path


class Model:
    def __init__(self, config):
        self.config = config
        self.label_dir = 'data/label.txt'
        self.vocal_dir = 'data/vocab.txt'
        self.categories, self.cat_to_id = read_category(self.label_dir)
        self.words, self.word_to_id = read_vocab(self.vocal_dir)

        conf = tf.ConfigProto()
        conf.gpu_options.allow_growth = True
        self.session = tf.Session(config=conf)
        self.session.run(tf.global_variables_initializer())
        tf.saved_model.loader.load(self.session,
                                   [tf.saved_model.tag_constants.TRAINING],
                                   get_model_file('model_path'))
        self.input_x = self.session.graph.get_tensor_by_name('input_x:0')
        self.keep_prob = self.session.graph.get_tensor_by_name('keep_prob:0')
        self.y_pred_cls = self.session.graph.get_tensor_by_name('output/predictions:0')

    def predict(self, content):
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]
        feed_dict = {
            self.input_x: pad_sequences([data], self.config.seq_length),
            self.keep_prob: 1.0
        }
        y_pred_cls = self.session.run(self.y_pred_cls, feed_dict=feed_dict)
        return self.categories[y_pred_cls[0]]


# def test():
#     session = tf.Session()
#     session.run(tf.global_variables_initializer())
#     saver = tf.train.Saver()
#     saver.restore(sess=session, save_path=save_path)  # 读取保存的模型
#     feed_dict = {
#         model.input_x: x_test[start_id:end_id],
#         model.keep_prob: 1.0
#     }
#     y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)


if __name__ == '__main__':
    input = input('please input CNN or RNN')
    test_demo = ['贷款可以用于买车吗',
                 '这钱我不借了，帮我取消']

    if input == 'CNN':
        print('predict CNN model...')
        model = Model(TCNNConfig)
        for i in test_demo:
            print(model.predict(i))

    elif input == 'RNN':
        print('predict RNN model...')
        model = Model(TRNNConfig)
        for i in test_demo:
            print(model.predict(i))

    else:
        print('input wrong,please input CNN or RNN')
