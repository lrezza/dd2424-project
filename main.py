import numpy as np
import sys
from matplotlib import pyplot
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.datasets import cifar10

def main():
    run_test_harness()

# run the test harness for evaluating a model
def run_test_harness():
    # load dataset
    (trainX, trainY), (testX, testY) = load_dataset()
    
    # define model
    model = define_model()
    # fit model, should eventually get validation data from training data
    history = model.fit(trainX, trainY, epochs=3, batch_size=64, validation_data=(testX, testY), verbose=1)
    # evaluate model
    _, acc = model.evaluate(testX, testY, verbose=0)
    print('> %.3f' % (acc * 100.0))
    # learning curves
    summarize_diagnostics(history)

def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# plot diagnostic learning curves
def summarize_diagnostics(history):
    pyplot.figure(figsize=(7, 8))
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    pyplot.legend()
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.legend()
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()

#load cifar10 dataset into a a train and test set
def load_dataset():
    (trainX, trainy), (testX, testy) = cifar10.load_data()

    trainY = to_categorical(trainy)
    testY = to_categorical(testy)

    trainX = normalize_data(trainX)
    testX = normalize_data(testX)

    return (trainX, trainY), (testX, testY)

#converts image data from unsigned ints in 0-255 to floats i 0-1 range
def normalize_data(X):
    X_norm = X.astype('float32')
    X_norm = X_norm / 255.0
    return X_norm

if __name__ == "__main__":
    main()