# 3. tagging means processes a sequence of words, and attaches a part of speech tag to each word

import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

# PunktSentenceTokenizer is unsupervised but it can be trained

sample_text = state_union.raw("2006-GWBush.txt")
train_text = state_union.raw("2005-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

print(sample_text)

def process_content():
    try:
        for i in tokenized:
                words = nltk.word_tokenize(i)
                tagged = nltk.pos_tag(words)
                print(tagged)

    except Exception as e:
        print(e)

# Here we see that and is CC, a coordinating conjunction; now and completely are RB, or adverbs;
# for is IN, a preposition; something is NN, a noun; and different is JJ, an adjective.
process_content()


