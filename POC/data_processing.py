import pickle
import re
import time
from os import listdir

import nltk
from nltk.corpus import wordnet

import config

# Stopword list
# nltk.download()
# print ("Downloading stopwords ......")
# nltk.download('stopwords')

stop_words = nltk.corpus.stopwords.words('english')
lmtzr = nltk.WordNetLemmatizer().lemmatize


# load files into memory
def load_files(filename):
    # open the file as read only
    file = open(filename, encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# split a document into news story and highlights
def split_data(doc):
    # find first headlines
    index = doc.find('@highlight')
    # split into story and headlines
    article, headlines = doc[:index], doc[index:].split('@highlight')
    # strip extra white space around each highlight
    headlines = [h.strip() for h in headlines if len(h) > 0]
    return article, headlines


def clean_text(lines, remove_stopwords=True):
    ''' clean a list of lines'''

    cleaned = list()
    # prepare a translation table to remove punctuation
    #table = str.maketrans(' ', ' ', string.punctuation)
    for line in lines:
        # strip source cnn office if it exists
        index = line.find('(CNN)  -- ')
        if index > -1:
            line = line[index + len('(CNN)'):]
        else:
            index = line.find('(CNN)')
            if index > -1:
                line = line[index + len('(CNN)'):]

        # tokenize on white space
        line = line.split()

        # convert to lower case
        line = [word.lower() for word in line]

        # Optionally, remove stop words
        if remove_stopwords:
            line = [w for w in line if w not in stop_words]

        # remove punctuation from each token
        #line = [w.translate(table) for w in line]

        # remove tokens with numbers in them
        line = [word for word in line if word.isalpha()]

        # Format words and remove unwanted characters
        text = " ".join(line)
        text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\<a href', ' ', text)
        text = re.sub(r'&amp;', '', text)
        text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
        text = re.sub(r'<br />', ' ', text)
        text = re.sub(r'\'', ' ', text)

        # remove empty strings
        if len(text )> 0 :
            cleaned.append(text)

    return cleaned


# load all stories in a directory
def load_stories(location):
    stories = list()
    file_list = listdir(location)
    total_files = len (file_list)
    count = 0
    print ("Total Files : {total_files}".format(total_files= total_files))
    clean_articles = []
    clean_headlines = []
    for name in file_list:
        count = count + 1
        filename = location + '/' + name
        # load document
        print('Loading  - {filename}, files number  - {count},  out of - {total_files}'
              .format(filename=filename, count=count, total_files=total_files))
        doc = load_files(filename)
        # split into story and highlights
        article, headlines = split_data(doc)
        # store
        #stories.append({'article': article, 'headlines' : headlines})

        article = clean_text(article.split('\n'))
        article = normalize_text(article)
        clean_articles.append(' '.join(article))
        headlines = clean_text(headlines, remove_stopwords=False)
        headlines = normalize_text(headlines)
        clean_headlines.append(' '.join(headlines))

    return clean_articles, clean_headlines


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def normalize_text(text):
    cleaned = list()

    for line in text :
        word_pos = nltk.pos_tag(nltk.word_tokenize(line))
        lemm_words = [lmtzr(sw[0], get_wordnet_pos(sw[1])) for sw in word_pos]

        word = [x.lower() for x in lemm_words]
        cleaned.append(' '.join(word))

    return cleaned


def main():
    start = time.perf_counter()
    clean_articles, clean_headlines = load_stories(config.path)
    print("Total Articles  : {len_articles} , Total Headlines : {len_headlines}- Time Taken : {time_taken}"
          .format(len_articles=len(clean_articles), len_headlines =len(clean_headlines), time_taken = (time.perf_counter()-start)/60))

    print ("Serialization of articles")
    # Store Articles (serialize)
    with open(config.base_path + config.articles_pickle_filename, 'wb') as handle:
        pickle.dump(clean_articles, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Serialization of headlines")
    # Store Articles (serialize)
    with open(config.base_path + config.headlines_pickle_filename, 'wb') as handle:
        pickle.dump(clean_headlines, handle, protocol=pickle.HIGHEST_PROTOCOL)


'''-------------------------main------------------------------'''
main()

''' FULL process takes 12 hours 15 mins'''
