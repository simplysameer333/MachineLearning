import time

import numpy as np
import tensorflow as tf

import build_model
import config
import vectorization

# This could later be improved as tensorflow provide that put padding by it owns.
def pad_sentence_batch(sentence_batch, vocab_to_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    padded_batch  = [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]
    # print ("padded  ==== > ", padded_batch)
    return padded_batch


def get_batches(headlines, articles, batch_size, vocab_to_int):
    """Batch headlines, articles, and the lengths of their sentences together"""
    for batch_i in range(0, len(articles) // batch_size):
        start_i = batch_i * batch_size
        headlines_batch = headlines[start_i:start_i + batch_size]
        articles_batch = articles[start_i:start_i + batch_size]
        pad_headlines_batch = np.array(pad_sentence_batch(headlines_batch, vocab_to_int))
        pad_articles_batch = np.array(pad_sentence_batch(articles_batch, vocab_to_int))

        # Need the lengths for the _lengths parameters
        pad_headlines_lengths = []
        for headline in pad_headlines_batch:
            pad_headlines_lengths.append(len(headline))

        pad_articles_lengths = []
        for article in pad_articles_batch:
            pad_articles_lengths.append(len(article))

        yield pad_headlines_batch, pad_articles_batch, pad_headlines_lengths, pad_articles_lengths


def train_model(train_graph, train_op, cost, gen_input_data, gen_targets, gen_lr, gen_keep_prob,
                gen_headline_length, gen_max_headline_length, gen_article_length,
                sorted_headlines_short, sorted_articles_short, vocab_to_int):
    # Record the update losses for saving improvements in the model
    headlines_update_loss = []

    # name given to checkpoint
    checkpoint = "best_model.ckpt"

    # This make sures that in one epoch it only checked as per value specified of per_epoch
    # e.g if length of article is 4000 the => 4000 / 32 (bath size) = > 125 (it means we will have 125 loops in 1 epoch)
    # then 125 / 3 - 1 = 40 (so while covering 125 iteartion per epoch after 40 iteration it will check and print the loss)
    update_check = (len(sorted_articles_short) // config.batch_size // config.per_epoch) - 1
    print("init value of update_check", update_check)

    with tf.Session(graph=train_graph) as sess:
        # This is to show graph in tensorboard
        # G:\Python\MLLearning\MachineLearning\POC > tensorboard --logdir = logs - -port 6006
        # TensorBoard 1.10.0 at http: // Sam: 6006(Press CTRL + C to quit)
        writer = tf.summary.FileWriter('logs', graph=sess.graph)
        sess.run(tf.global_variables_initializer())

        # If we want to continue training a previous session
        # loader = tf.train.import_meta_graph("./" + checkpoint + '.meta')
        # loader.restore(sess, checkpoint)

        for epoch_i in range(1, config.epochs + 1):
            update_loss = 0
            batch_loss = 0
            for batch_i, (headlines_batch, articles_batch, headlines_lengths, articles_lengths) in enumerate(
                    get_batches(sorted_headlines_short, sorted_articles_short, config.batch_size, vocab_to_int)):
                print("batch_i ==== ", batch_i)
                start_time = time.time()
                _, loss = sess.run(
                    [train_op, cost],
                    {gen_input_data: articles_batch,
                     gen_targets: headlines_batch,
                     gen_lr: config.learning_rate,
                     gen_headline_length: headlines_lengths,
                     gen_article_length: articles_lengths,
                     gen_keep_prob: config.keep_probability})

                batch_loss += loss
                update_loss += loss
                end_time = time.time()
                batch_time = end_time - start_time

                # This prints status after value specified in display_step. Helps to to see progress
                if batch_i % config.display_step == 0 and batch_i > 0:
                    print('Epoch {}/{} Batch {}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                          .format(epoch_i,
                                  config.epochs,
                                  batch_i,
                                  len(sorted_articles_short) // config.batch_size,
                                  batch_loss / config.display_step,
                                  batch_time * config.display_step))
                    batch_loss = 0

                # print loss value after after steps specified in update_check
                if batch_i % update_check == 0 and batch_i > 0:
                    print("Average loss for this update:", round(update_loss / update_check, 3))
                    headlines_update_loss.append(update_loss)

                    # If the update loss is at a new minimum, save the model
                    if update_loss <= min(headlines_update_loss):
                        print('New Record!')
                        stop_early = 0
                        saver = tf.train.Saver()
                        saver.save(sess, checkpoint)

                    else:
                        print("No Improvement.")
                        stop_early += 1
                        if stop_early == config.stop:
                            break
                    update_loss = 0

            # Reduce learning rate, but not below its minimum value
            learning_rate *= config.learning_rate_decay
            if learning_rate < config.min_learning_rate:
                learning_rate = config.min_learning_rate

            if stop_early == config.stop:
                print("Stopping Training.")
                break


def main():
    print("Prepare input parameters ...")
    sorted_articles, sorted_headlines, vocab_to_int, word_embedding_matrix = vectorization.create_input_for_graph()
    print("Build Graph parameters ...")
    train_graph, train_op, cost, gen_input_data, gen_targets, gen_lr, gen_keep_prob, gen_headline_length, gen_max_headline_length, \
    gen_article_length = build_model.build_graph(vocab_to_int, word_embedding_matrix)

    # Subset the data for training, this is used to check if steps are working fine.
    # In actual run whole data should be taken
    start = config.start
    end = start + 4000

    print("Total Articles that we have for this run :", len(sorted_articles))
    # Train the Model
    sorted_headlines_short = sorted_headlines[start:end]
    sorted_articles_short = sorted_articles[start:end]
    print("Total Articles samples taken for this run :", len(sorted_articles_short))
    print("The shortest text length:", len(sorted_articles_short[0]))
    print("The longest text length:", len(sorted_articles_short[-1]))

    train_model(train_graph, train_op, cost, gen_input_data, gen_targets, gen_lr, gen_keep_prob,
                gen_headline_length, gen_max_headline_length, gen_article_length,
                sorted_headlines_short, sorted_articles_short, vocab_to_int)


'''-------------------------main------------------------------'''
main()
