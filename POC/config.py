#base_path = 'G:\\AI\\data\\cnn\\'
base_path = '../data/cnn/'
path = base_path + 'stories/sample/'
stories_pickle_filename = "news.pickle"

# to avoid words that are used less that threshold value
threshold = 5
# Dimension size as per pre-trained data
embedding_dim = 300

max_text_length = 84
max_summary_length = 13
min_length = 2
unk_text_limit = 1
