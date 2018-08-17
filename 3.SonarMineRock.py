import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def multi_layer_perceptron(x, weights, biases):
    layer1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer1 = tf.nn.sigmoid(layer1)

    layer2 = tf.add(tf.matmul(layer1, weights['h2']), biases['b2'])
    layer2 = tf.nn.sigmoid(layer2)

    layer3 = tf.add(tf.matmul(layer2, weights['h3']), biases['b3'])
    layer3 = tf.nn.sigmoid(layer3)

    layer4 = tf.add(tf.matmul(layer3, weights['h4']), biases['b4'])
    layer4 = tf.nn.sigmoid(layer4)

    output_layer = tf.add(tf.matmul(layer4, weights['out']), biases['out'])
    return output_layer

def one_hot_encode(label):
    n_label = len(label)
    n_unique_label = len(np.unique(label))

    one_hot_encode = np.zeros((n_label, n_unique_label))
    one_hot_encode[np.arange(n_label), label] = 1

    return one_hot_encode, n_unique_label


def read_dataset():
    path = '..\data\sonar.csv'
    df = pd.read_csv(path)
    x = df[df.columns[0:60]].values
    y = df[df.columns[60]]

    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    y, unique_labels = one_hot_encode(y)
    return x,y, unique_labels

X, y, unique_classes = read_dataset()
X, y = shuffle(X, y, random_state = 1)

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)

# define hyper parameter -> fixed paramter not like (w, or b)
learning_rate = 0.3
learning_epochs = 100 # iteration to improve
cost_history = np.empty(shape=[1], dtype = float)
n_dim = X.shape[1]
print('n_dim  - {n_dim} '.format(n_dim=n_dim))
print('unique_labels  - {unique_labels} '.format(unique_labels=unique_classes))

n_hidden_1 = 60
n_hidden_2 = 60
n_hidden_3 = 60
n_hidden_4 = 60

# columns are fixed =n_dim , rows are None hence dynamic. SO that weight can be assign to each column
x = tf.placeholder(tf.float32, [None, n_dim])

# This is for matrix mutliplication -> rows ar multiplied with column
w = tf.Variable(tf.zeros(([n_dim, unique_classes])))

# This is because as of now if we multiple x and x then i would return None  and  unique_classes (matrix mutliplication)
b = tf.Variable(tf.zeros(( unique_classes)))

# output (column is set none and output can only b 2
y_ = tf.placeholder(tf.float32, [None,  unique_classes])

# These are assigned as we matrix multiplication
weights = {
    'h1' : tf.Variable(tf.truncated_normal([n_dim, n_hidden_1])),
    'h2' : tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
    'h3' : tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
    'h4' : tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])),
    'out' : tf.Variable(tf.truncated_normal([n_hidden_4, unique_classes]))
}

biases = {
    'b1' : tf.Variable(tf.truncated_normal([n_hidden_1])),
    'b2' : tf.Variable(tf.truncated_normal([n_hidden_2])),
    'b3' : tf.Variable(tf.truncated_normal([n_hidden_3])),
    'b4' : tf.Variable(tf.truncated_normal([n_hidden_4])),
    'out' : tf.Variable(tf.truncated_normal([unique_classes]))
}

init = tf.global_variables_initializer()
y = multi_layer_perceptron(x, weights, biases)

# check this (cost between y and y_)
cost_fact = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_))

training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_fact)

# standard session (used in earlier example can also be used)
sess = tf.InteractiveSession()
sess.run(init)

# Mean squared error
mse_history = []
accuracy_history = []

for epoch in range (learning_epochs):
    sess.run(training_step, feed_dict = {x : train_X, y_ : train_y})
    cost = sess.run(cost_fact, feed_dict = {x : train_X, y_ : train_y})
    cost_history = np.append(cost_history, cost)
    correct_prediction = tf.equal(tf.argmax(y, 1),  tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    prediction_y = sess.run(y, feed_dict= {x : test_X})
    mse = tf.reduce_mean(tf.square(prediction_y - test_y))

    mse_ = sess.run(mse)
    mse_history .append(mse_)
    accuracy = sess.run(accuracy, feed_dict= {x: train_X, y_:train_y})
    accuracy_history.append(accuracy)

    print('epoch - {epoch} , cost - {cost},  MSE - {mse},  training accuracy - {t_accuracy}'
          .format(epoch=epoch, cost= cost, mse = mse_, t_accuracy=accuracy))


plt.plot(mse_history, 'r')
plt.show()

plt.plot(accuracy_history, 'r')
plt.show()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('test Accuracy  - {accuracy} '.format(accuracy = sess.run(accuracy, feed_dict={x:test_X, y_: test_y})))

pred_y = sess.run(y, feed_dict={x:test_X})
mse = tf.reduce_mean(tf.square(pred_y - test_y))

print('Last mse  - {mse} '.format(mse=sess.run(mse)))








