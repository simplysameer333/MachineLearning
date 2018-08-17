import tensorflow as tf


weight = tf.Variable([0.5], tf.float32)
var_a = tf.Variable([-0.1], tf.float32)

x_place = tf.placeholder(tf.float32)
model = weight * x_place + var_a
y_place = tf.placeholder(tf.float32)

# loss function (diff is actual value and prediction)
squared_diff = tf.square(model - y_place)
loss = tf.reduce_sum(squared_diff)

# default is usually -> 0.01, it should change the different values with increment/decrement of 0.01 (steps of 0.01)
optimizer = tf.train.GradientDescentOptimizer(0.01)
training = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    output = sess.run(loss, {x_place:[1,2,3] , y_place:[5,6,7]})
    print(output)
    for i in range (100): # to get better w, a increase range 100 to 200 or 300
        sess.run(training, {x_place:[1,2,3] , y_place:[5,6,7]})
        print (sess.run([weight, var_a]))
    output = sess.run(loss, {x_place: [1, 2, 3], y_place: [5, 6, 7]})
    print(output)
