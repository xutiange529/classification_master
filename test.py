import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICE'] = '0'

import pandas as pd
from predict import Model
from cnn_model import TCNNConfig
from rnn_model import TRNNConfig
from sklearn.metrics import confusion_matrix, classification_report


def single_test(input):
    if input == 'CNN':
        print('predict CNN model...')
        model = Model(TCNNConfig)
    else:
        print('predict RNN model...')
        model = Model(TRNNConfig)

    print(model.predict('贷款我就不要了'))


def batch_test():
    if input == 'CNN':
        print('predict CNN model...')
        model = Model(TCNNConfig)
    else:
        print('predict RNN model...')
        model = Model(TRNNConfig)

    data = pd.read_excel('data/test.xlsx')
    x = list(data['text_a'])
    y_pred = []
    for text_a in x:
        tmp = model.predict(text_a)
        y_pred.append(tmp)
    y_pred = pd.DataFrame(y_pred, columns=['pred'])
    res = pd.concat([data, y_pred], axis=1)
    res['judge'] = res['label'] == res['pred']
    res.to_excel('data/res.xlsx', index=False)

    label = res['label']
    pred = res['pred']
    label_list = ['有经营场所', '无经营场所', '高危回答', '无效回答']

    matrix = confusion_matrix(label, pred, labels=label_list)
    matrix = pd.DataFrame(matrix, columns=label_list)
    print(matrix)
    matrix.to_excel('data/confusion_matrix.xlsx', index=False)

    report = classification_report(label, pred)
    print(report)
    report = to_table(report)
    report.to_excel('data/report.xlsx', index=False)


def to_table(report):
    report = report.splitlines()
    res = []
    res.append([''] + report[0].split())
    for row in report[2:-2]:
        res.append(row.split())
    lr = report[-1].split()
    res.append([''.join(lr[:3])] + lr[3:])
    return pd.DataFrame(res)


if __name__ == '__main__':
    input = input('please input CNN or RNN')
    single_test(input)

    batch_test()
