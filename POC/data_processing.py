import nltk
import pickle
import string
from gensim.models.keyedvectors import KeyedVectors
import config
#from pycontractions import Contractions

# nltk.download()
# print ("Downloading stopwords ......")
# nltk.download('stopwords')

# Stopword list
stop_words = nltk.corpus.stopwords.words('english')


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


def vectorization (lines):
    '''Using FB enlish word2Vec pre-trained model'''
    #model = KeyedVectors.load_word2vec_format('G:\Python\MLLearning\MachineLearning\data\wiki.en.vec', binary=False)
    model = KeyedVectors.load_word2vec_format('G:\Python\MLLearning\MachineLearning\data\wiki-news-300d-1M.vec', binary=False)

    vectors_set = list()
    for line in lines:
        # if you vector file is in binary format, change to binary=True
        try:
            word = [w for w in line]
            vectors_set.append(model[word])
        except KeyError:
            print (word + " not in vocabulary")
            c = 0
    return vectors_set


def missing_word_ratio(word_counts, embeddings_index) :
    # Find the number of words that are missing from CN, and are used more than our threshold.
    missing_words = 0
    threshold = 20

    for word, count in word_counts.items():
        if count > threshold:
            if word not in embeddings_index:
                missing_words += 1

    missing_ratio = round(missing_words / len(word_counts), 4) * 100
    return missing_ratio, missing_words


def main():
    # Load data (deserialize)
    print ("Loading data ......")
    with open(config.base_path+config.stories_pickle_filename, 'rb') as handle:
        all_stories = pickle.load(handle)

    # clean stories
    word_counts = {}
    for example in all_stories:
        example['article'] = clean_lines(example['article'].split('\n'))
        example['headlines'] = clean_lines(example['headlines'], remove_stopwords=False)

    print (all_stories[1])

    count_words(word_counts, example['article'])
    print("Size of Vocabulary:", len(word_counts))

    embeddings_index = list();
    print ("creating embedding index .....")
    for example in all_stories:
        embeddings_index.append(vectorization(example['article']))

    print('Word embeddings:', len(embeddings_index))

    # find out missing words and thr %
    missing_ratio, missing_words = missing_word_ratio(word_counts, embeddings_index)

    print("Number of words missing from CN:", missing_words)
    print("Percent of words that are missing from vocabulary: {}%".format(missing_ratio))



'''------- Read main ----------'''
main()
