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
__all__ = [Sequential, Dense, LSTM]
numpy.random.seed(7)


top_words=5000
(X_train,y_train),(X_test,y_test) = imdb.load_data(num_words=top_words)
max_review_length=500
X_train=sequence.pad_sequences(X_train,maxlen=max_review_length)
X_test=sequence.pad_sequences(X_test,maxlen=max_review_length)

embedding_vector_length=32
model=Sequential()
model.add(Embedding(top_words,embedding_vector_length,input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())
model.fit(X_train,y_train,epochs=3,batch_size=64)
scores=model.evaluate(X_test,y_test,verbose=0)
print("Accuracy:%.2f%%" % (scores[1]*100))