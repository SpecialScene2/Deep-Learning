import tensorflow as tf

import numpy as np
from numpy import genfromtxt

import os

import matplotlib.pyplot as plt
import matplotlib


if "DISPLAY" not in os.environ:
    # remove Travis CI Error
    matplotlib.use('Agg')

def normalization_2d(data):
    for i in range(len(data)):
        for j in range(6):
            data[i][j] = (data[i][j]-raw_mean)/raw_std
    print("normalization_2d :",data)

def normalization(data):
    for i in range(len(data)):
        data[i] = (data[i]-raw_mean)/raw_std
    print("normalization:",data)

def denormalization(data):
    for i in range(len(data)):
         data[i] = (data[i] * raw_std)+raw_mean

def denormalization_2d(data):
    for i in range(len(data)):
        for j in range(6):
            data[i][j] = (data[i][j] * raw_std)+raw_mean
    print(data)

def predict(file):
    train = genfromtxt('train.csv', delimiter=',', skip_header=1, usecols=(1, 2, 3, 4, 5, 6))
    label = genfromtxt('train.csv', delimiter=',', skip_header=1, usecols=(7))
    raw = genfromtxt('train.csv', delimiter=',', skip_header=1, usecols=(1))

    # 예측해야할 test
    test = genfromtxt('train.csv', delimiter=',', skip_header=1, usecols=(1))
    # 예측한 결과를 담아야 할 result (test에서 구한거 denormalization 해야 함.)


    # raw Data로 정규화를 위한 평균 및 표준 편차를 구한 식
    raw_mean = np.mean(raw)
    raw_std = np.std(raw)
    print(np.mean(raw), np.std(raw))

    # normalization
    normalization_2d(train)
    normalization(label)

    # Data Split (train : 70%, test : 30 %) /
    # IDEA : Hyper-parameter 정할 땐 valid set 활용 하고 최종 학습 할 땐 train 100 다 사용 하기.
    train_size = int(len(train) * 0.7)
    print("Train_Size :", train_size)
    test_size = len(train) - train_size

    trainX, validX = np.array(train[0:train_size]), np.array(train[train_size:len(train)])
    trainY, validY = np.array(label[0:train_size]), np.array(label[train_size:len(train)])

    # train Parameters
    seq_length = 6  # 6일이라서 seq_length는 6이다.
    data_dim = 1  # 6일의 통으로 되어있는 data가 온도라는 피쳐 1개로 이루어져있음을 뜻함.
    # 이전 hidden_dim : 100
    hidden_dim = 600
    output_dim = 1
    learning_rate = 0.0001
    # 이전 iterations = 1000
    iterations = 2000

    # Data 6일 / Label 1일로 나누기
    # 1) Train Dataset
    # build a Dataset for correct shape
    trainX_Li = []
    trainY_Li = []
    for i in range(0, len(trainY) - seq_length):
        _x = trainX[i]
        _y = trainY[i + seq_length]
        #     print(_x, "->", _y)
        trainX_Li.append(_x)
        trainY_Li.append(_y)

    trainX_Li = np.array(trainX_Li)
    trainX_Li = trainX_Li.reshape(-1, 6, 1)

    trainY_Li = np.array(trainY_Li)
    trainY_Li = trainY_Li.reshape(-1, 1)

    # 2) Valid Dataset
    # build a Valid Dataset for correct shpae
    validX_Li = []
    validY_Li = []
    for i in range(0, len(validY) - seq_length):
        _x = validX[i]
        _y = validY[i + seq_length]  # Next close price
        #     print(_x, "->", _y)
        validX_Li.append(_x)
        validY_Li.append(_y)

    validX_Li = np.array(validX_Li)
    validX_Li = validX_Li.reshape(-1, 6, 1)

    validY_Li = np.array(validY_Li)
    validY_Li = validY_Li.reshape(-1, 1)

    #NN Setting
    X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
    Y = tf.placeholder(tf.float32, [None, 1])

    # build a LSTM network
    cells = []
    dropout = tf.placeholder(tf.float32)
    # 이전 range(5)
    for _ in range(3):
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
        #     cell = tf.contrib.rnn.GRUCell(num_units=hidden_dim)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0 - 0.3)
        cells.append(cell)
    # print(cells)
    cell = tf.contrib.rnn.MultiRNNCell(cells)

    outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    # We use the last cell's output
    Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], output_dim, activation_fn=None)

    # cost/loss
    loss = tf.reduce_sum(tf.square(Y - Y_pred))  # sum of the squares
    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    targets = tf.placeholder(tf.float32, [None, 1])  # valid_Y가 들어감
    predictions = tf.placeholder(tf.float32, [None, 1])  # valid_X가 들어감
    rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        # Training step
        for i in range(iterations):
            _, step_loss = sess.run([train, loss], feed_dict={
                X: trainX_Li, Y: trainY_Li})
            print("[step: {}] loss: {}".format(i, step_loss))

        # Test step
        test_predict = sess.run(Y_pred, feed_dict={X: validX_Li})
        rmse_val = sess.run(rmse, feed_dict={
            targets: validY_Li, predictions: test_predict})
        print("RMSE: {}".format(rmse_val))

        # # 과제 제출용 step
        # test_predict = sess.run(Y_pred, feed_dict={X: validX_Li})
        # rmse_val = sess.run(rmse, feed_dict={
        #     targets: validY_Li, predictions: test_predict})
        # print("RMSE: {}".format(rmse_val))

        plt.figure(figsize=(60, 40))
        plt.plot(validY_Li, color="black")
        plt.plot(test_predict, color="blue")
        plt.xlabel("Time Period")
        plt.ylabel("Temperature")
        plt.show()
    return list([10.0 for _ in range(52)])


def write_result(predictions):
    # You don't need to modify this function.
    with open('result.csv', 'w') as f:
        f.write('Value\n')
        for l in predictions:
            f.write('{}\n'.format(l))


def main():
    # You don't need to modify this function.
    predictions = predict('test.csv')
    write_result(predictions)

# def MinMaxScaler(data):
#     numerator = data - np.min(data, 0)
#     denominator = np.max(data, 0) - np.min(data, 0)
#     # noise term prevents the zero division
#     return numerator / (denominator + 1e-7)


if __name__ == '__main__':
    # You don't need to modify this part.
    main()

filename = "test.csv"
predict(filename)