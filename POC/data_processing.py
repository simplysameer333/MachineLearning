import nltk
import pickle
import string
from gensim.models.keyedvectors import KeyedVectors
import config
import numpy as np
import time
#from pycontractions import Contractions

# nltk.download()
# print ("Downloading stopwords ......")
# nltk.download('stopwords')

# Stopword list
stop_words = nltk.corpus.stopwords.words('english')

# Load Model
''' https://fasttext.cc/docs/en/english-vectors.html '''
print ("Loading Pre-Trained Model ..... " )
start = time.perf_counter()
model = KeyedVectors.load_word2vec_format('G:\Python\MLLearning\MachineLearning\data\wiki-news-300d-1M.vec', binary=False)
# model = KeyedVectors.load_word2vec_format('C:\Temp\python_files\MLLearning\data\wiki-news-300d-1M.vec', binary=False)
print ("Loaded Pre-Trained Model, time taken",  ((time.perf_counter()  - start)/60))


def clean_lines(lines, remove_stopwords = True):
    '''clean a list of lines'''

    cleaned = list()
    # prepare a translation table to remove punctuation
    table = str.maketrans('', '', string.punctuation)
    for line in lines:
        # strip source cnn office if it exists
        index = line.find('(CNN)  -- ')
        if index > -1:
            line = line[index+len('(CNN)'):]
        else:
            index = line.find('(CNN)')
            if index > -1:
                line = line[index + len('(CNN)'):]

        # tokenize on white space
        line = line.split()

        # convert to lower case
        line = [word.lower() for word in line]

        # remove punctuation from each token
        line = [w.translate(table) for w in line]

        # remove tokens with numbers in them
        line = [word for word in line if word.isalpha()]

        # Optionally, remove stop words
        if remove_stopwords:
            line = [w for w in line if not w in stop_words]

        cleaned.append(' '.join(line))

    # remove empty strings
    cleaned = [c for c in cleaned if len(c) > 0]
    return cleaned


def count_words(count_dict, text):
    ''' Count the number of occurrences of each word in a set of text'''
    for sentence in text:
        for word in sentence.split():
            if word not in count_dict:
                count_dict[word] = 1
            else:
                count_dict[word] += 1


def vectorization (text, embeddings_index):
    for sentence in text:
        try:
            for vocab_word in sentence.split():
                embeddings_index[vocab_word] = model[vocab_word]
                #print("Work : {vocab_word} , vector value : {vector_value}".format(vocab_word=vocab_word, vector_value =vector_value))
        except KeyError:
            print("{vocab_word} not in vocabulary".format(vocab_word=vocab_word))
            #embeddings_index[vocab_word] = -1


def missing_word_ratio(word_counts, embeddings_index) :
    # Find the number of words that are missing from CN, and are used more than our threshold.
    missing_words_count = 0
    missing_words = list()

    for word, count in word_counts.items():
        #found_dim = embeddings_index.get(word, -1)
        #print("{found_dim} is found_dim ".format(found_dim=found_dim))
        #if found_dim == -1 and word not in missing_words:
        if word not in embeddings_index and word not in missing_words :
            missing_words_count += 1
            missing_words.append(word)
            print("{word} is missing ".format(word=word))

    missing_ratio = round(missing_words_count / len(word_counts), 4) * 100
    return missing_ratio, missing_words_count


def main():
    # Load data (deserialize)
    print ("Loading data ......")
    with open(config.base_path+config.stories_pickle_filename, 'rb') as handle:
        all_stories = pickle.load(handle)

    # clean stories
    for example in all_stories:
        example['article'] = clean_lines(example['article'].split('\n'))
        example['headlines'] = clean_lines(example['headlines'], remove_stopwords=False)

    #print (all_stories[1])
    word_counts = {}
    for example in all_stories:
        count_words(word_counts, example['article'])

    print("Size of Vocabulary:", len(word_counts))

    embeddings_index = {};
    print ("creating embedding index .....")
    for example in all_stories:
        vectorization(example['article'], embeddings_index)

    print('Word embeddings:', len(embeddings_index))

    # find out missing words and thr %
    missing_ratio, missing_words_count = missing_word_ratio(word_counts, embeddings_index)

    print("Number of words missing :", missing_words_count)
    print("Percent of words that are missing from vocabulary: {}%".format(missing_ratio))

'''------- Read main ----------'''
main()
