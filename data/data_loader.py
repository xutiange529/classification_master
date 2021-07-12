# coding: utf-8

import sys
import jieba
import pandas as pd
import numpy as np
from collections import Counter
from cnn_model import TCNNConfig
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


def remove_chars(text):
    remove_list = [' ', ',', '。', '？']
    for char in remove_list:
        text = str(text).replace(char, '')
    return text


def preprocess_data():
    data = pd.read_excel('data/data.xlsx')
    label_list = data['label'].unique().tolist()
    if len(label_list) == TCNNConfig().num_classes:
        print('labels right')
    else:
        print('labels wrong')

    data['text_a'] = data['text_a'].apply(lambda x: remove_chars(x))
    df_train, df_valid, y_train, y_valid = train_test_split(
        data, data['label'], test_size=0.2, stratify=data['label'], random_state=1,
    )
    df_train.to_excel('data/train.xlsx', index=False)
    df_valid.to_excel('data/val.xlsx', index=False)


def read_file(filename):
    """读取文件数据"""
    data = pd.read_excel(filename)
    contents = list(data['text_a'])
    labels = list(data['label'])
    return contents, labels


def build_char_vocab(train_dir, vocab_dir, vocab_size=10000):
    """根据训练集构建词汇表，存储"""
    contents, _ = read_file(train_dir)
    all_data = []
    for content in contents:
        all_data.extend(str(content))

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open(vocab_dir, mode='w').write('\n'.join(words) + '\n')


def build_vocab(train_dir, vocab_dir, vocab_size=10000):
    """根据训练集构建词汇表，存储"""
    contents, _ = read_file(train_dir)
    all_data = []
    for content in contents:
        generator = jieba.cut(str(content), cut_all=False)
        content = ' '.join(generator)
        all_data.extend(str(content).split(' '))

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    words = list(words)
    words.sort(key=lambda i: len(i), reverse=True)
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open(vocab_dir, mode='w').write('\n'.join(words) + '\n')


def read_vocab(vocab_dir):
    """读取词汇表"""
    with open(vocab_dir, 'r', encoding='utf-8') as f:
        words = [_.strip() for _ in f.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_category(label_dir):
    """读取分类目录，固定"""
    with open(label_dir, 'r', encoding='gbk') as f:
        dict_ = eval(f.read())
        categories = list(dict_)
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id


def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)
    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in str(contents[i]) if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])
    x_pad = pad_sequences(data_id, max_length)
    y_pad = to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示
    return x_pad, y_pad


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1
    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    for x in sequences:
        try:
            lengths.append(len(x))
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break
    # is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    # if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
    #     raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
    #                      "You should set `dtype=object` for variable length strings."
    #                      .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x
