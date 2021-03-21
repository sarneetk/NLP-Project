# CNN for the IMDB problem
from tensorflow import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from util import f1_score, recall, precision, plot_graph

PLOT_GRAPH = False
PLOT_MODEL = False

# load the dataset but only keep the top 5000 words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# pad dataset to a maximum review length in words
max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)
# Define CNN  Model
# first layer is the Embedded layer that uses 32 length vectors to represent each word.
# The next layer is the one dimensional CNN layer .
# Finally, because this is a classification problem we use a Dense output layer with a single neuron and
# a sigmoid activation function to make 0 or 1 predictions for the two classes (good and bad) in the problem.
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_words))
model.add(Conv1D(32, 3, padding='same', activation='relu'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1_score, precision, recall])
model.summary()
# Fit the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=128, verbose=2)

# Evaluation of the model with training data
scores_train = model.evaluate(X_train, y_train, verbose=0)
print("Training Data: ")
print("Accuracy: %.2f%%, F_1Score: %.2f%% , Precision: %.2f%%, Recall: %.2f%% " % (scores_train[1]*100, scores_train[2]*100,
                                                                                   scores_train[3]*100, scores_train[4]*100))

# Evaluation of the model with test data
scores = model.evaluate(X_test, y_test, verbose=0)
print("Test Data:")
print("Accuracy: %.2f%%, F_1Score: %.2f%% , Precision: %.2f%%, Recall: %.2f%%" % (scores[1] * 100, scores[2] * 100,
                                                                                 scores[3] * 100, scores[4] * 100))

if PLOT_GRAPH:
    plot_graph(history)

if PLOT_MODEL:
    img_file = 'model_diagrams/cnn.png'
    keras.utils.plot_model(model, to_file=img_file, show_shapes=True, show_layer_names=True)