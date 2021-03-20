# CNN for the IMDB problem
from tensorflow import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras import backend as K
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


def recall(y_true, y_pred):
    y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    if (true_positives!=0):
        recall0 = true_positives / (all_positives + K.epsilon())
    else:
        recall0=0.0
    return recall0

def precision(y_true, y_pred):
    y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    if(true_positives!=0):
        precision0 = true_positives / (predicted_positives + K.epsilon())
    else:
        precision0=0.0
    return precision0

def f1_score(y_true, y_pred):
    precision1 = precision(y_true, y_pred)
    recall1 = recall(y_true, y_pred)
    return 2* ((precision1 * recall1) / (precision1 + recall1 + K.epsilon()))

if __name__ == '__main__':
    # load the dataset but only keep the top 5000 words, zero the rest
    top_words = 5000
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
    # pad dataset to a maximum review length in words
    max_review_length = 500
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
    # Define CNN  Model
    # first layer is the Embedded layer that uses 32 length vectors to represent each word.
    # The next layer is the one dimensional CNN layer .
    # Finally, because this is a classification problem we use a Dense output layer with a single neuron and
    # a sigmoid activation function to make 0 or 1 predictions for the two classes (good and bad) in the problem.
    # Define LSTM Model
    # first layer is the Embedded layer that uses 32 length vectors to represent each word.
    # The next layer is the LSTM layer with 32 memory units (smart neurons).
    # Finally, because this is a classification problem we use a Dense output layer with a single neuron and
    # a sigmoid activation function to make 0 or 1 predictions for the two classes (good and bad) in the problem.
    embedding_vector_length = 32
    model = Sequential()
    model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    model.add(LSTM(32))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',f1_score, precision, recall])

    model.summary()
    # Fit the model
    history=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=128, verbose=2)

    # Evaluation of the model with training data
    scores_train = model.evaluate(X_train, y_train, verbose=0)
    print("Training Data: ")
    print("Accuracy: %.2f%%, F_1Score: %.2f%% , Precision: %.2f%%, Recall: %.2f%% " % (scores_train[1]*100,scores_train[2]*100,
                                                                                       scores_train[3]*100,scores_train[4]*100))

    # Evaluation of the model with test data
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Test Data:")
    print("Accuracy: %.2f%%, F_1Score: %.2f%% , Precision: %.2f%%, Recall: %.2f%%" % (scores[1] * 100,scores[2] * 100 ,
                                                                                     scores[3] * 100,scores[4] * 100))
    # Plotting the graph
    plot_graph(history)