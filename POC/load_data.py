from os import listdir
import time
import pickle
import config


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


# load all stories in a directory
def load_stories(location):
    stories = list()
    file_list = listdir(location)
    total_files = len (file_list)
    count = 0
    print ("Total Files : {total_files}".format(total_files= total_files))
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
        stories.append({'article': article, 'headlines' : headlines})
    return stories


# load articles
start = time.process_time()
all_stories = load_stories(config.path)
print ("Time Taken : {time} mins".format(time = (time.process_time()  - start)/60))
print('Loaded Stories %d' % len(all_stories))

# Store data (serialize)
with open(config.base_path+config.stories_pickle_filename, 'wb') as handle:
    pickle.dump(all_stories, handle, protocol=pickle.HIGHEST_PROTOCOL)


