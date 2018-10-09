import tensorflow as tf
from tensorflow.python.layers.core import Dense

import config
import vectorization

print('TensorFlow Version: {}'.format(tf.__version__))

# Getting the Hyperparameters
epochs = config.epochs
batch_size = config.batch_size
rnn_size = config.rnn_size
num_layers = config.num_layers
learning_rate = config.learning_rate
keep_probability = config.keep_probability


def model_inputs():
    '''Create palceholders for inputs to the model'''

    input_data = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    article_length = tf.placeholder(tf.int32, (None,), name='article_length')
    max_article_length = tf.reduce_max(article_length, name='max_dec_len')
    headline_length = tf.placeholder(tf.int32, (None,), name='headline_length')

    return input_data, targets, lr, keep_prob, article_length, max_article_length, headline_length


def encoding_layer(rnn_size, article_length, num_layers, rnn_inputs, keep_prob):
    '''Create the encoding layer'''

    # Number of layer inside neural network
    for layer in range(num_layers):
        with tf.variable_scope('encoder_{}'.format(layer)):
            # forward direction cell with random weights with seed value for reproduce random value
            cell_fw = tf.contrib.rnn.LSTMCell(rnn_size,
                                              initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))

            # Dropout to kills cells that are not changing.
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw,
                                                    input_keep_prob=keep_prob)

            cell_bw = tf.contrib.rnn.LSTMCell(rnn_size,
                                              initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw,
                                                    input_keep_prob=keep_prob)

            # Bidirectional as it si more optimized, spl with Dropouts
            enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                    cell_bw,
                                                                    rnn_inputs,
                                                                    article_length,
                                                                    dtype=tf.float32)
    # Join outputs since we are using a bidirectional RNN
    enc_output = tf.concat(enc_output, 2)

    return enc_output, enc_state


def process_encoding_input(target_data, vocab_to_int, batch_size):
    '''Remove the last word id from each batch and concat the <GO> to the begining of each batch'''
    # sample = [[11, 12, 13], [31, 32, 33], [51, 52, 53], [61, 62,63]]
    # slice = tf.strided_slice(sample, begin=[0,0], end=[4,4], strides=[1,1])
    # process_input = tf.concat([tf.fill([4, 1], 9999), slice], 1)
    # process_input = [[9999   11   12   13], [9999   31   32   33] , [9999   51   52   53], [9999   61   62   63]]

    # target data has batch_size rows, -1 means everything, so first elect of each row
    slice = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])

    # tf.fill creates array of batch_size X 1 and then fill in value of '<GO>'
    # create matrix that has first column as value vocab_to_int['<GO>'] and second as index [first column of each row)
    process_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), slice], 1)

    return process_input


def training_decoding_layer(dec_embed_input, article_length, dec_cell, initial_state, output_layer,
                            max_headline_length):
    '''Create the training logits'''

    training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                        sequence_length=article_length,
                                                        time_major=False)

    training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                       training_helper,
                                                       initial_state,
                                                       output_layer)

    training_logits, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder, impute_finished=True,
                                                              maximum_iterations=max_headline_length)

    return training_logits


def inference_decoding_layer(embeddings, start_token, end_token, dec_cell, initial_state, output_layer,
                             max_article_length, batch_size):
    '''Create the inference logits'''

    start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [batch_size], name='start_tokens')
    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,
                                                                start_tokens,
                                                                end_token)

    inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                        inference_helper,
                                                        initial_state,
                                                        output_layer)

    inference_logits, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                            output_time_major=False,
                                                            impute_finished=True,
                                                            maximum_iterations=max_article_length)

    return inference_logits


def decoding_layer(dec_embed_input, embeddings, enc_output, enc_state, vocab_size, headline_length, article_length,
                   max_article_length, rnn_size, vocab_to_int, keep_prob, batch_size, num_layers):
    '''Create the decoding cell and attention for the training and inference decoding layers'''

    for layer in range(num_layers):
        with tf.variable_scope('decoder_{}'.format(layer)):
            lstm = tf.contrib.rnn.LSTMCell(rnn_size,
                                           initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            dec_cell = tf.contrib.rnn.DropoutWrapper(lstm,
                                                     input_keep_prob=keep_prob)

    output_layer = Dense(vocab_size,
                         kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

    attn_mech = tf.contrib.seq2seq.BahdanauAttention(rnn_size,
                                                     enc_output,
                                                     headline_length,
                                                     normalize=False,
                                                     name='BahdanauAttention')

    dec_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell,
                                                   attn_mech,
                                                   rnn_size)
    #
    # initial_state = tf.contrib.seq2seq.DynamicAttentionWrapperState(enc_state[0],
    #                                                                _zero_state_tensors(rnn_size,
    #                                                                                    batch_size,
    #                                                                                    tf.float32))

    initial_state = dec_cell.zero_state(batch_size=batch_size, dtype=tf.float32).clone(cell_state=enc_state)

    with tf.variable_scope("decode"):
        training_logits = training_decoding_layer(dec_embed_input,
                                                  article_length,
                                                  dec_cell,
                                                  initial_state,
                                                  output_layer,
                                                  vocab_size,
                                                  max_article_length)
    with tf.variable_scope("decode", reuse=True):
        inference_logits = inference_decoding_layer(embeddings,
                                                    vocab_to_int['<GO>'],
                                                    vocab_to_int['<EOS>'],
                                                    dec_cell,
                                                    initial_state,
                                                    output_layer,
                                                    max_article_length,
                                                    batch_size)

    return training_logits, inference_logits


def seq2seq_model(input_data, target_data, keep_prob, headline_length, article_length, max_article_length,
                  vocab_size, rnn_size, num_layers, vocab_to_int, batch_size, word_embedding_matrix):
    '''Use the previous functions to create the training and inference logits'''

    # Use fasttext's embeddings and the newly created ones as our embeddings
    embeddings = word_embedding_matrix

    # embedding_lookup returns embedding values of input_data that we have provided
    enc_embed_input = tf.nn.embedding_lookup(embeddings, input_data)

    # Define encoder layers - with respect to size of neurons, hidden layers and design (such as bi-directional)
    enc_output, enc_state = encoding_layer(rnn_size, article_length, num_layers, enc_embed_input, keep_prob)

    # creating put for input to decoder - at start it will have INT value of 'GO' to tell that new statement has arrived.
    dec_input = process_encoding_input(target_data, vocab_to_int, batch_size)

    # embedding_lookup returns embedding values of input_data that we have provided, this time for decoder input
    dec_embed_input = tf.nn.embedding_lookup(embeddings, dec_input)

    print("getting decoding_layer logits ... ")
    training_logits, inference_logits = decoding_layer(dec_embed_input,
                                                       embeddings,
                                                       enc_output,
                                                       enc_state,
                                                       vocab_size,
                                                       headline_length,
                                                       article_length,
                                                       max_article_length,
                                                       rnn_size,
                                                       vocab_to_int,
                                                       keep_prob,
                                                       batch_size,
                                                       num_layers)

    return training_logits, inference_logits


def build_graph(vocab_to_int, word_embedding_matrix):
    # Build the graph
    train_graph = tf.Graph()
    # Set the graph to default to ensure that it is ready for training
    with train_graph.as_default():
        # Load the model inputs
        print("Load input parameter ...")
        input_data, targets, lr, keep_prob, article_length, max_article_length, headline_length = model_inputs()

        # Create the training and inference logits
        print("Create instance of seq2seq model parameter ...")
        seq2seq_model(tf.reverse(input_data, [-1]),
                      targets,
                      keep_prob,
                      headline_length,
                      article_length,
                      max_article_length,
                      len(vocab_to_int) + 1,
                      rnn_size,
                      num_layers,
                      vocab_to_int,
                      batch_size,
                      word_embedding_matrix)


def main():
    print("Prepare input parameters ...")
    vocab_to_int, word_embedding_matrix = vectorization.create_input_for_graph()
    print("Build Graph parameters ...")
    build_graph(vocab_to_int, word_embedding_matrix)


'''-------------------------main------------------------------'''
main()
