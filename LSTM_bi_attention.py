import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Layer
from keras.layers import LSTM, Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import keras.backend as K
from util import f1_score, recall, precision

__all__ = [Sequential, Dense, LSTM]

# initializing the random number generator to a constant value to ensure we can easily reproduce the results.
numpy.random.seed(7)


# create an attention class (wrapper on top of Layer)
class Attention(Layer):
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences
        super(Attention, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, x):
        et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        at = K.softmax(et)
        at = K.expand_dims(at, axis=-1)
        output = x * at

        if self.return_sequences:
            return output

        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super(Attention, self).get_config()


# load the IMDB dataset.
# We are constraining the dataset to the top 5,000 words.
# We also split the dataset into train (50%) and test (50%) sets.
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# truncate and pad the input sequences so that they are all the same length for modeling.
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# Define LSTM Model
# first layer is the Embedded layer that uses 32/64 length vectors to represent each word.
# The next layer is the LSTM layer with 100 memory units (smart neurons).
# Finally, because this is a classification problem we use a Dense output layer with a single neuron and
# a sigmoid activation function to make 0 or 1 predictions for the two classes (good and bad) in the problem.
embedding_vector_length = 64
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model.add(Bidirectional(LSTM(50, return_sequences=True)))
model.add(Attention(return_sequences=True))  # receive 3D and output 3D
model.add(LSTM(50))
model.add(Dense(1, activation='sigmoid'))
# log loss is used as the loss function (ADAM optimization algorithm).
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1_score, precision, recall])
print(model.summary())
# A large batch size of 64 reviews is used to space out weight updates.
model.fit(X_train, y_train, epochs=3, batch_size=64)

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
