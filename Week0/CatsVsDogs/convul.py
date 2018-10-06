import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

import pickle
import time

global x_train,y_train,x_train_norm

def load_processed_data():
    global x_train,y_train,x_train_norm
    NAME="Cats vs Dogs-cnn".format(int(time.time()))
    tensorboard=TensorBoard(log_dir='logs/{}'.format(NAME))
    pickle_in = open("x_train.pickle","rb")
    x_train = pickle.load(pickle_in)

    pickle_in = open("y_train.pickle","rb")
    y_train = pickle.load(pickle_in)

    x_train_norm=x_train/255.0



def CNN():
    global x_train,y_train,x_train_norm
    model = Sequential()

    model.add(Conv2D(256, (3, 3),input_shape=x_train_norm.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(64))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    model.fit(x_train_norm, y_train, batch_size=32, epochs=10, validation_split=0.1)

    model.save('Cats_Dogs.model')
    print("Training Completed and Saved")





print('Loading preprocessed data')
load_processed_data()






print("Training Model")
CNN()
