
import numpy
import tensorflow as tf
# from tf.keras.datasets import imdb
from tensorflow import keras
# from keras import datasets
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import matplotlib.pyplot as plt
plt.style.use('ggplot')

__all__ = [Sequential, Dense, LSTM]

def  plot_graph(history) :
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # initializing the random number generator to a constant value to ensure we can easily reproduce the results.
    numpy.random.seed(7)

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
    # first layer is the Embedded layer that uses 32 length vectors to represent each word.
    # The next layer is the LSTM layer with 100 memory units (smart neurons).
    # Finally, because this is a classification problem we use a Dense output layer with a single neuron and
    # a sigmoid activation function to make 0 or 1 predictions for the two classes (good and bad) in the problem.
    embedding_vector_length = 32
    model = Sequential()
    model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    model.add(LSTM(100,dropout=0.2,recurrent_dropout=0.2))

    model.add(Dense(250, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # log loss is used as the loss function (ADAM optimization algorithm).
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    # A large batch size of 64 reviews is used to space out weight updates.
    #model.fit(X_train, y_train, epochs=1, batch_size=64)
    history=model.fit(X_train, y_train, batch_size=64, epochs=15, verbose=1,  validation_data=(X_test, y_test))  # starts training

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print(scores)
    print("Accuracy:%.2f%%" % (scores[1] * 100))
    plot_graph(history)

