#import nltk

# nltk.set_proxy('http://applicationwebproxy.nomura.com:80')
# nltk.download()

# 1. Tokenization => Words and sentence

# Corporas - body of text - speeches, articles

# Lexicon - Dictionary - words and meanings
# e.g = Bull -> someone positive about market
# english = Bull-  an animal

from nltk.tokenize import sent_tokenize, word_tokenize
example_text = "Had Mr. Stokes been found guilty, he not only could have faced a jail sentence - " \
               "the charge of affray carries a maximum jail term of three years in the UK - but " \
               "the possibility of a long ban from the game. England's World Cup and Ashes plans could " \
               "have been in jeopardy. Stokes' career and reputation certainly were."

print ("sentence ======")
print (sent_tokenize(example_text))

print ("Words ======")
print (word_tokenize(example_text))

# 2. Stop words
# Words does not have any meaning, they are filler and not required for data analysis
# e.g -> a, the etc
# by using this - >we remove unwanted data from whole text string

from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

# print ("english stop_words ======")
# print(stop_words)

words = word_tokenize(example_text)
filtered_sent = []

for w in words:
    if w not in stop_words:
        filtered_sent.append(w)

print ("text after filter stop_words ======")
print(filtered_sent)

# 3. Stemming means or words. eg. reading -> read
# as most of the time they does not help
# e.g -> 1. I was taking ride on the horse. 2. I was riding the horse
# In this case they both means same


from nltk.stem import PorterStemmer

ps = PorterStemmer()

example_words = ["python", "pythoner", "pythoning", "pythoned", "pythonly"]
for w in example_words:
    print(ps.stem(w))
