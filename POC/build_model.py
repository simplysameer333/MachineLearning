import tensorflow as tf
from tensorflow.python.layers.core import Dense

import config
import vectorization

# Getting the Hyperparameters
epochs = config.epochs
batch_size = config.batch_size
rnn_size = config.rnn_size
num_layers = config.num_layers
learning_rate = config.learning_rate
keep_probability = config.keep_probability


def model_inputs():
    '''Create palceholders for inputs to the model'''

    input_data = tf.placeholder(tf.int32, [None, None], name='input_data')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    headline_length = tf.placeholder(tf.int32, (None,), name='headline_length')
    max_headline_length = tf.reduce_max(headline_length, name='max_headline_length')
    article_length = tf.placeholder(tf.int32, (None,), name='article_length')

    return input_data, targets, lr, keep_prob, headline_length, max_headline_length, article_length


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


def encoding_layer(rnn_size, article_length, num_layers, rnn_inputs, keep_prob):
    '''Create the encoding layer'''

    # Number of layer inside neural network
    for layer in range(num_layers):
        with tf.variable_scope('encoder_{}'.format(layer)):

            # forward direction cell with random weights with seed value for reproduce random value
            cell_fw = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))

            # Dropout to kills cells that are not changing.
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=keep_prob)

            cell_bw = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=keep_prob)

            # Bidirectional as it is more optimized, spl with Dropouts
            enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, rnn_inputs, article_length, dtype=tf.float32)

    # Join outputs since we are using a bidirectional RNN
    enc_output = tf.concat(enc_output, 2)

    return enc_output, enc_state


def train_decoding_layer(dec_embed_input, headline_length, dec_cell, initial_state, output_layer, max_headline_length):
    '''Create the training logits'''

    # for training : read inputs from dense ground truth vector
    training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                        sequence_length=headline_length,
                                                        time_major=False)

    training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                       training_helper,
                                                       initial_state,
                                                       output_layer)

    training_logits, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder, impute_finished=True,
                                                              output_time_major=False,
                                                              maximum_iterations = max_headline_length)

    return training_logits


def inference_decoding_layer(embeddings, start_token, end_token, dec_cell, initial_state, output_layer,
                             max_headline_length, batch_size):
    '''Create the inference logits'''

    start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [batch_size], name='start_tokens')
    '''
    # For Basic decoder
    # GreedyEmbeddingHelper - > Select top probability output
    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,
                                                                start_tokens,
                                                                end_token)

    inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                        inference_helper,
                                                        initial_state,
                                                        output_layer)

    inference_logits, _, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                               output_time_major=False,
                                                               impute_finished=True,
                                                               maximum_iterations=max_headline_length)
    '''
    beam_initial_state = dec_cell.zero_state(config.batch_size * config.beam_width, tf.float32)

    inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
        cell=dec_cell,
        embedding=embeddings,
        start_tokens=start_tokens,
        end_token=end_token,
        initial_state=beam_initial_state,
        beam_width=config.beam_width,
        output_layer=output_layer,
        length_penalty_weight=0.0)

    inference_logits, _, _ = tf.contrib.seq2seq.dynamic_decode(
        decoder=inference_decoder,
        impute_finished=False,
        maximum_iterations=2 * max_headline_length)

    return inference_logits


def decoding_layer(dec_embed_input, embeddings, enc_output, enc_state, vocab_size,  article_length, headline_length,
                   max_headline_length, rnn_size, vocab_to_int, keep_prob, batch_size, num_layers):
    '''Create the decoding cell and attention for the training and inference decoding layers'''

    # creating layer and Dropout layers
    for layer in range(num_layers):
        with tf.variable_scope('decoder_{}'.format(layer)):
            lstm = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            dec_cell = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)

    # creating Dense- This is also called output layer. This will produce the summary.
    output_layer = Dense(vocab_size, activation='relu', kernel_initializer=
    tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

    # Using BahdanauAttention as one of the widely used Attention Algorithms
    attn_mech = tf.contrib.seq2seq.BahdanauAttention(rnn_size, enc_output, article_length,
                                                     normalize=False, name='BahdanauAttention')

    dec_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell, attn_mech, rnn_size)

    # tf.contrib.seq2seq.AttentionWrapperState(
    #     cell_state=enc_state,
    #     time=tf.zeros([], dtype=tf.int32),
    #    attention=_zero_state_tensors(rnn_size, batch_size, tf.float32),
    #    alignments=self._attention_mechanism.initial_alignments(
    #        batch_size, dtype),
    #    alignment_history=alignment_history)


    # initial_state = tf.contrib.seq2seq.DynamicAttentionWrapperState(enc_state[0],  _zero_state_tensors(rnn_size,
    #                                                                                   batch_size,
    #                                                                                 tf.float32))
    # alignment_history = ()
    # attention_state = ()
    # initial_state = tf.contrib.seq2seq.AttentionWrapperState( cell_state=enc_state[0],
    #                                                                time=tf.zeros([], dtype=tf.int32),
    #                                                                attention=_zero_state_tensors(rnn_size, batch_size,
    #                                                                                              tf.float32),
    #                                                                alignments=attn_mech.initial_alignments(
    #                                                                    batch_size, tf.float32),
    #                                                                alignment_history=alignment_history,
    #                                                                attention_state = attention_state
    #                                                                )

    # initializing the initial state, layer it would be update by output from one cell
    # initial_state = dec_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
    initial_state = dec_cell.zero_state(batch_size=batch_size, dtype=enc_state[0].dtype).clone(cell_state=enc_state[0])

    # Creating training logits - which would be used during training dataset
    with tf.variable_scope("decode"):
        training_logits = train_decoding_layer(dec_embed_input,
                                               headline_length,
                                               dec_cell,
                                               initial_state,
                                               output_layer,
                                               max_headline_length)

    # Creating inference logits - which would produce output using train model
    with tf.variable_scope("decode", reuse=True):
        inference_logits = inference_decoding_layer(embeddings,
                                                    vocab_to_int['<GO>'],
                                                    vocab_to_int['<EOS>'],
                                                    dec_cell,
                                                    initial_state,
                                                    output_layer,
                                                    max_headline_length,
                                                    batch_size)

    return training_logits, inference_logits



def seq2seq_model(input_data, target_data, keep_prob, article_length, headline_length, max_headliney_length,
                      vocab_size, rnn_size, num_layers, vocab_to_int, batch_size, word_embedding_matrix):
    '''Use the previous functions to create the training and inference logits'''

    # Use fasttext's embeddings and the newly created ones as our embeddings
    embeddings = word_embedding_matrix

    # embedding_lookup returns embedding values of input_data that we have provided
    print("Geting embedding for encoder input")
    enc_embed_input = tf.nn.embedding_lookup(embeddings, input_data)

    # Define encoder layers - with respect to size of neurons, hidden layers and design (such as bi-directional)
    print("Initializing encoder layers")
    enc_output, enc_state = encoding_layer(rnn_size, article_length, num_layers, enc_embed_input, keep_prob)

    print("Adding 'GO' to start text")
    dec_input = process_encoding_input(target_data, vocab_to_int, batch_size)

    print("Getting embedding for encoder input")
    dec_embed_input = tf.nn.embedding_lookup(embeddings, dec_input)

    print("Getting decoding_layer logits ... ")
    # Train: Learn model parameters.
    # Inference: Apply model on unseen data to assess performance.
    training_logits, inference_logits = decoding_layer(dec_embed_input,
                                                       embeddings,
                                                       enc_output,
                                                       enc_state,
                                                       vocab_size,
                                                       article_length,
                                                       headline_length,
                                                       max_headliney_length,
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
        input_data, targets, lr, keep_prob, headline_length, max_headline_length, article_length = model_inputs()

        # Create the training and inference logits
        print("Create instance of seq2seq model parameter ...")

        # training_logits gives us matrix of possibilities when we trained the system whereas
        # inference_logits are used when we are trying to predict summary out of it.
        training_logits, inference_logits = seq2seq_model(tf.reverse(input_data, [-1]),
                                                          targets,
                                                          keep_prob,
                                                          article_length,
                                                          headline_length,
                                                          max_headline_length,
                                                          len(vocab_to_int) + 1,
                                                          rnn_size,
                                                          num_layers,
                                                          vocab_to_int,
                                                          batch_size,
                                                          word_embedding_matrix)

        # Create tensors for the training logits and inference logits
        training_logits = tf.identity(training_logits.rnn_output, 'logits')

        # inference_logits would be used while predicting the summary
        # used for basic decoder
        # inference_logits = tf.identity(inference_logits.sample_id, name='predictions')
        inference_logits = tf.identity(inference_logits.predicted_ids, name='predictions')

        # Create the weights for sequence_loss
        masks = tf.sequence_mask(headline_length, max_headline_length, dtype=tf.float32, name='masks')

        with tf.name_scope("optimization"):
            # Loss function
            cost = tf.contrib.seq2seq.sequence_loss(training_logits, targets, masks)

            # Optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate)

            # Gradient Clipping
            gradients = optimizer.compute_gradients(cost)
            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients)
    print("Graph is built.")
    # input_data, targets, lr, keep_prob, headline_length, max_headline_length, article_length
    return train_graph, train_op, cost, input_data, targets, lr, keep_prob, headline_length, max_headline_length, \
           article_length

def main():
    print('TensorFlow Version: {}'.format(tf.__version__))  # we are using 1.10
    print ("Prepare input parameters ...")
    sorted_articles, sorted_headlines, vocab_to_int, word_embedding_matrix = vectorization.create_input_for_graph()
    print("Build Graph parameters ...")
    build_graph(vocab_to_int, word_embedding_matrix)


'''-------------------------main------------------------------'''
#main ()
