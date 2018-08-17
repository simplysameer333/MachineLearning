import tensorflow as tf

# TF Constants
node_first = tf.constant(10.0, tf.float32)
node_second = tf.constant(20.0, tf.float32)
division = node_first/node_second

# using 'with' no need to close session
with tf.Session() as sess:
    output = sess.run([division])
    print (output)

# TF Placeholder
place_a = tf.placeholder(tf.float32)
place_b = tf.placeholder(tf.float32)
addition = place_a + place_b

with tf.Session() as sess:
    output = sess.run(addition, {place_a: [10,20], place_b:20})
    print(output)


# TF variable
weight = tf.Variable([0.5], tf.float32)
var_a = tf.Variable([-0.1], tf.float32)

x_place = tf.placeholder(tf.float32)
model = weight * x_place + var_a
y_place = tf.placeholder(tf.float32)

# loss function (diff is actual value and prediction)
squared_diff = tf.square(model - y_place)
loss = tf.reduce_sum(squared_diff)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    output = sess.run(loss, {x_place:[1,2,3] , y_place:[5,6,7]})
    print(output)

