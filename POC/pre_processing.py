import nltk
import pickle
import string


#nltk.download()
nltk.download('stopwords')

# Stopword list
stop_words = nltk.corpus.stopwords.words('english')

base_path = 'G:\\AI\\data\\cnn\\'
path = base_path + 'stories\\\sample\\'
stories_pickle_filename = "news.pickle"

'''clean a list of lines'''


def clean_lines(lines, remove_stopwords = True):
    cleaned = list()
    # prepare a translation table to remove punctuation
    table = str.maketrans('', '', string.punctuation)
    for line in lines:
        # strip source cnn office if it exists
        index = line.find('(CNN)  -- ')
        if index > -1:
            line = line[index+len('(CNN)'):]
        # tokenize on white space
        line = line.split()
        # convert to lower case
        line = [word.lower() for word in line]
        # remove punctuation from each token
        line = [w.translate(table) for w in line]
        # remove tokens with numbers in them
        # line = [word for word in line if word.isalpha()]
        # store as string
        # Optionally, remove stop words
        if remove_stopwords:
            line = [w for w in line if not w in stop_words]
        cleaned.append(' '.join(line))
    # remove empty strings
    cleaned = [c for c in cleaned if len(c) > 0]
    return cleaned


# Load data (deserialize)
with open(base_path+stories_pickle_filename, 'rb') as handle:
    all_stories = pickle.load(handle)

print (all_stories[1])

# clean stories
for example in all_stories:
    example['article'] = clean_lines(example['article'].split('\n'))
    example['headlines'] = clean_lines(example['headlines'], remove_stopwords=False)

print (all_stories[1])