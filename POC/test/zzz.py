import tensorflow as tf

sample = tf.constant(
    [[11, 12, 13],
     [31, 32, 33],
     [51, 52, 53],
     [61, 62, 63]])

print(sample)

slice = tf.strided_slice(sample, begin=[0, 0], end=[4, 4], strides=[1, 1])
process_input = tf.concat([tf.fill([4, 1], 9999), slice], 1)

with tf.Session() as sess:
    print(sess.run(process_input))

print('Epoch {}/{} Batch {}'.format(1000000, 12, 111))

update_check = (4538 // 32 // 3) - 1
print("init value of update_check", update_check)
