#-- coding: utf-8 --
#@Time : 2021/4/1 11:05
#@Author : HUANG XUYANG
#@Email : xhuang032@e.ntu.edu.sg
#@File : Logistic_Regression.py
#@Software: PyCharm

import numpy as np
import sklearn.datasets as sk_dataset

train_batch_size = 32
iteration = 500
learning_rate = 0.01
cp = 10


class LRClassifier:
    """A simple logistic regression.

    Choose one of sgd or bdg as optimizer.
    The learning rate is gradually reduced.

    Attributes:
        theta: A numpy array, [n_feature, 2].
        lr: A floating number of learning rate.
        gradient_decent: The gradient decent method you choose.
        data_std: A list, store normalization parameters for each layer.
        data_mean: A list, store normalization parameters for each layer.
        same_feature: A bool, the true means all the features have same meaning and boundary for example: images.

    """
    def __init__(self, all_data, optimizer='sgd', lr=0.001, same_feature=False):
        n_feature = len(all_data[0])
        self.theta = np.random.random([n_feature, 2])
        self.lr = lr
        self.gradient_decent = getattr(self, optimizer)
        if same_feature is True:
            self.data_std = np.std(all_data)
            self.data_mean = np.mean(all_data)
        else:
            self.data_std = np.std(all_data, axis=0)
            self.data_mean = np.mean(all_data, axis=0)

    def train(self, data, label):
        """

        :param data: Training data.
        :param label: Training label.
        :return: No return
        """

        assert len(data.shape) > 1, 'Data shape should be [n, dim].'
        assert len(data) == len(label), 'Label number does not match data number.'
        assert len(label.shape) == 1, 'Label should be 1-D array.'

        label = self.one_hot(label, 2)
        data = self.standardize(data)
        y_pred = self.sigmoid(np.dot(data, self.theta))
        self.gradient_decent(data, y_pred, label)

    def sigmoid(self, x):
        return 1 / (1 + np.e ** (-x))

    def predict(self, data, output_prob=False):
        """

        :param data: Predict data.
        :param output_prob: A bool number, if True return the raw predict probability, if False return predict class.
        :return: Prediction result.
        """

        data = self.standardize(data)
        y_pred = self.sigmoid(np.dot(data, self.theta))
        if output_prob is False:
            y_pred = np.argmax(y_pred, axis=1)
        return y_pred

    def eval(self, data, label):
        """

        :param data: Evaluation data.
        :param label: Evaluation label.
        :return: Accuracy.
        """

        assert len(data.shape) > 1, 'Data shape should be [n, dim].'
        assert len(data) == len(label), 'Label number does not match data number.'
        assert len(label.shape) == 1, 'Label should be 1-D array.'

        data = self.standardize(data)
        y_pred = self.sigmoid(np.dot(data, self.theta))
        y_pred = np.argmax(y_pred, axis=1)
        acc = np.sum(np.equal(y_pred, label)) / len(label)
        return acc

    def one_hot(self, x, n_class):
        y = np.zeros([len(x), n_class])
        for i in range(len(x)):
            y[i, x[i]] = 1
        return y

    def standardize(self, x):
        return (x - self.data_mean) / self.data_std

    def bgd(self, x, y_pred, y_true):
        self.theta -= self.lr * np.dot(x.transpose(), y_pred - y_true)

    def sgd(self, x, y_pred, y_true):
        rand_index = np.random.randint(0, len(y_pred))
        y_pred = y_pred[rand_index]
        y_true = y_true[rand_index]
        x = x[rand_index]
        self.theta -= self.lr * np.dot(x[np.newaxis, :].transpose(), (y_pred - y_true)[np.newaxis, :])


def prepare_data(proportion):
    dataset = sk_dataset.load_breast_cancer()
    label = dataset['target']
    data = dataset['data']
    n_class = len(dataset['target_names'])

    shuffle_index = np.arange(len(label))
    np.random.shuffle(shuffle_index)

    train_number = int(proportion * len(label))
    train_index = shuffle_index[:train_number]
    val_index = shuffle_index[train_number:]
    data_train = data[train_index]
    label_train = label[train_index]
    data_val = data[val_index]
    label_val = label[val_index]
    return (data_train, label_train), (data_val, label_val), n_class


def batch(x, y, batch_size):
    index = np.random.randint(0, len(x), len(x))
    np.random.shuffle(index)
    x = x[index]
    y = y[index]
    n_batch = int(np.floor(len(x)/batch_size))
    x_batch = [x[i*batch_size: (i+1)*batch_size] for i in range(n_batch)]
    y_batch = [y[i*batch_size: (i+1)*batch_size] for i in range(n_batch)]
    if n_batch*batch_size != len(x):
        x_batch.append(x[n_batch*batch_size:])
        y_batch.append(y[n_batch*batch_size:])
    return iter(x_batch), iter(y_batch)


if __name__ == '__main__':
    train, val, num_class = prepare_data(0.9)
    lr_classifier = LRClassifier(train[0], optimizer='sgd', lr=learning_rate)

    for i in range(iteration):
        train_data, train_label = batch(train[0], train[1], batch_size=train_batch_size)
        lr_classifier.lr = learning_rate * (1 - (i/iteration)) ** cp
        for data_batch, label_batch in zip(train_data, train_label):
            lr_classifier.train(data_batch, label_batch)
        print('train acc:', lr_classifier.eval(train[0], train[1]))
        print('val acc:', lr_classifier.eval(val[0], val[1]))

