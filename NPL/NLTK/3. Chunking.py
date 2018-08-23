#4. Chunking
# find out noun(place, person, thing) [also called entity].  then find out words that have modifers effect on noun.
# one sentence might have many nouns, so its becomes importance to find relationship between noun and thr modifiers

import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

# PunktSentenceTokenizer is unsupervised but it can be trained

sample_text = state_union.raw("2006-GWBush.txt")
train_text = state_union.raw("2005-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)
#print(sample_text)

chuncked =[]
def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            chunkGram = r"""Chunk : {<RB.?>*<VB.?>*<NNP>+<NN>} """
            chunkParser = nltk.RegexpParser(chunkGram)
            chuncked = chunkParser.parse(tagged)
            #print (chuncked)

        chuncked.draw()
    except Exception as e:
        print(e)

# Here we see that and is CC, a coordinating conjunction; now and completely are RB, or adverbs;
# for is IN, a preposition; something is NN, a noun; and different is JJ, an adjective.
process_content()
