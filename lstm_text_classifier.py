# Code based on:
# https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/ and
# https://richliao.github.io/supervised/classification/2016/12/26/textclassifier-RNN/

import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical

# fix random seed for reproducibility
np.random.seed(7)

def clean_str(string):
    # Minimal string cleaning for text data
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    # Every dataset is lower cased
    return string.strip().lower()

data_train = pd.read_csv('data/labeledTrainData.tsv', sep='\t')

# read text data to sequences
texts = []
labels = []

for idx in range(data_train.sentiment.shape[0]):
    text = BeautifulSoup(data_train.review[idx], "lxml")
    texts.append(clean_str(text.get_text().encode('ascii', 'ignore')))
    labels.append(data_train.sentiment[idx])

# maximum number of words to keep, based on word frequency
MAX_NB_WORDS = 20000

# Tokenization
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# maximal length of sequence
MAX_SEQUENCE_LENGTH = 1000

# pad input sequences
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# converts a class vector (integers) to binary class matrix.
labels = to_categorical(np.asarray(labels))

# shuffle data
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# training/testing data split
VALIDATION_SPLIT = 0.1
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
x_train = data[:data.shape[0]-nb_validation_samples]
y_train = labels[:data.shape[0]-nb_validation_samples]
x_test = data[data.shape[0]-nb_validation_samples:]
y_test = labels[data.shape[0]-nb_validation_samples:]

print('Number of positive and negative reviews in traing and validation set ')
print y_train.sum(axis=0)
print y_test.sum(axis=0)

# LSTM model.

# The first layer is the Embedded layer that uses 32 length vectors to represent each word.
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, embedding_vecor_length, input_length=MAX_SEQUENCE_LENGTH))

# The next layer is the LSTM layer with 100 neurons.
model.add(LSTM(100))

# Finally, because this is a classification problem we use a Dense output layer and a sigmoid activation function
# to make 0 or 1 predictions for the two classes (good and bad) in the problem.
model.add(Dense(2, activation='sigmoid'))

# plot neural net architecture
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.eps', show_shapes=True, show_layer_names=True)

# Because it is a binary classification problem, log loss is used as the loss function.
# The efficient ADAM optimization algorithm is used.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# A large batch size of 64 reviews is used to space out weight updates.
# The model is fit for 2 epochs because it quickly overfits the problem.
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=2, batch_size=64)

# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
