import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import keras
from sklearn.utils import class_weight
import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl
import os
import seaborn as sns
from time import time
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

os.chdir('C:\\Users\\Tijev\\OneDrive\\Documenten\\School\\Machine Learning')

MNIST_train = np.load('written_train.npy')
MNIST_train = MNIST_train /255
MNIST_train = MNIST_train.reshape((-1,28,28,1))
MNIST_test = np.load('written_test.npy')
MNIST_test = MNIST_test/255
MNIST_test = MNIST_test.reshape((-1,28,28,1))

RNN_train = np.load('spoken_train.npy', allow_pickle=True)
RNN_test = np.load('spoken_test.npy', allow_pickle=True)


match_train = np.load('match_train.npy')
match_train_temp = ~match_train
match_train_test = np.stack((match_train, match_train_temp), axis=-1)


def padding(mfcc): #padding for the trainset
    output = np.empty(shape=(45000,93,13))
    for id, value in enumerate(mfcc):
        npad = 93 - len(value)
        output[id] = np.pad(value, pad_width=((0,npad),(0,0)), mode='constant')
    return output

def padding2(mfcc): #padding for test set
    output = np.empty(shape=(15000,93,13))
    for id, value in enumerate(mfcc):
        npad = 93 - len(value)
        output[id] = np.pad(value, pad_width=((0,npad),(0,0)), mode='constant')
    return output

RNN_train = padding(RNN_train)
RNN_test = padding2(RNN_test)
RNN_traincopy = RNN_train


MNIST_true = MNIST_train[match_train]
RNN_true = RNN_train[match_train]
length = len(MNIST_true)

MNIST_false = MNIST_train[~match_train]
RNN_false = RNN_train[~match_train]

#making a CNN for the MNIST dataset and a RNN for the spoken dataset
inputCNN = tf.keras.Input(shape=(28, 28, 1))
CNN = tf.keras.layers.Conv2D(32, (3,3), activation='relu')(inputCNN)
CNN = tf.keras.layers.BatchNormalization()(CNN)
CNN = tf.keras.layers.Dropout(0.2)(CNN)
CNN = tf.keras.layers.MaxPooling2D(2,2)(CNN)
CNN = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(CNN)
CNN = tf.keras.layers.BatchNormalization()(CNN)
#CNN = tf.keras.layers.Dropout(0.2)(CNN)
#CNN = tf.keras.layers.MaxPooling2D(2,2)(CNN)
#CCN = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(CNN)
#CNN = tf.keras.layers.BatchNormalization()(CNN)
#CNN = tf.keras.layers.Dropout(0.2)(CNN)
CNN = tf.keras.layers.MaxPooling2D(2,2)(CNN)
CNN = tf.keras.layers.Flatten()(CNN)

modelCNN = tf.keras.Model(inputs=inputCNN, outputs=CNN)

inputRNN = tf.keras.Input(shape=(93,13))
RNN = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(inputRNN)
RNN = tf.keras.layers.BatchNormalization()(RNN)
RNN = tf.keras.layers.Dropout(0.2)(RNN)
RNN = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False))(RNN)
RNN = tf.keras.layers.BatchNormalization()(RNN)
modelRNN = tf.keras.Model(inputs=inputRNN, outputs=RNN)

""" now we combine the layers """
combined = tf.keras.layers.concatenate([modelCNN.output, modelRNN.output])
final_dense = tf.keras.layers.Dense(64, activation='relu')(combined)
final_dense = tf.keras.layers.Dropout(0.2)(final_dense)
final_dense = tf.keras.layers.Dense(32, activation='relu')(combined)
final_dense = tf.keras.layers.Dense(10, activation='relu')(final_dense)
final_dense = tf.keras.layers.Dense(1, activation='sigmoid')(final_dense)

final_model = tf.keras.Model(inputs=[modelCNN.input, modelRNN.input], outputs=final_dense)


tensorboard = TensorBoard(log_dir='tensorboard/{}'.format(time()), histogram_freq=1, write_grads=True)

final_model.compile(optimizer='Adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

checkpoint = ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5')

sample_weights = np.ones(45000) #creating a sample_weight to give the True more value
sample_weights[match_train] = 12 

history = final_model.fit([MNIST_train, RNN_train], match_train, validation_split = 0.1, epochs= 50, batch_size= 32, sample_weight= sample_weights, callbacks=[tensorboard, checkpoint], shuffle=True)
# final_model.save('model_save2.h5')
predictions = final_model.predict([MNIST_test, RNN_test], verbose=1)

#plotting model accuracy and model loss at every epoch
print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#plotting predictions
pred_list = []

#turning predictions into results
for row in predictions:
    if row < 0.5:
        pred_list.append(0)
    elif row > 0.5:
        pred_list.append(1)
#saving results
bool_list = list(map(bool,pred_list))
results = np.asarray(bool_list)
np.save('result_simple_0.04.npy', results)


#plotting a confusion matrix to see the predicted labels
conf_matrix = confusion_matrix(match_train, pred_list, labels=None, sample_weight=None)
#
ax= plt.subplot()
sns.heatmap(conf_matrix, annot=True, ax = ax); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');