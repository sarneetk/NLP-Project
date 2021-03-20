# CNN for the IMDB problem
from tensorflow import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import matplotlib.pyplot as plt
plt.style.use('ggplot')

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
    # load the dataset but only keep the top 5000 words, zero the rest
    top_words = 5000
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
    # pad dataset to a maximum review length in words
    max_words = 5000
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
    model.add(Conv1D(64, 3, padding='same', activation='relu'))
    model.add(MaxPooling1D())
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    # Hidden Layers
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    #Outputlayer Layers
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    # Fit the model
    history=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=128, verbose=2)
    # Final evaluation of the model
    scores_train = model.evaluate(X_train, y_train, verbose=0)
    print("Training-Accuracy: %.2f%%" % (scores_train[1]*100))
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Test-Accuracy: %.2f%%" % (scores[1]*100))
    # Plot the graph
    plot_graph(history)

    img_file = '../model_diagrams/CNN_hidden_dropout_base.png'
    keras.utils.plot_model(model, to_file=img_file, show_shapes=True,  show_layer_names=True)

