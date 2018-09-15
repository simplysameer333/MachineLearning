base_path = 'G:\\AI\\data\\cnn\\'
#base_path = '..\\data\\cnn\\'
path = base_path + 'sample_5k\\'
#path = base_path + 'stories\\'
articles_pickle_filename = "articles.pickle"
headlines_pickle_filename = "headlines.pickle"
model_pickle_filename = "model.pickle"
word_embedding_matrix_filename = "word_embedding_matrix.pickle"

''' https://fasttext.cc/docs/en/english-vectors.html '''
model_path ='G:\Python\MLLearning\MachineLearning\data\wiki-news-300d-1M.vec'
# model_path= 'C:\Temp\python_files\MLLearning\data\wiki-news-300d-1M.vec'

# to avoid words that are used less that threshold value
threshold = 2

# Dimension size as per pre-trained data
embedding_dim = 300
max_text_length = 1000
max_summary_length = 20
min_length = 2
unk_text_limit = 200

# Set the Hyperparameters
epochs = 100
batch_size = 64
rnn_size = 256
num_layers = 2
learning_rate = 0.005
keep_probability = 0.75