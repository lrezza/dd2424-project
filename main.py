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
from keras.layers import Dropout  # Import Dropout layer
from keras.optimizers import SGD
from keras.datasets import cifar10

def main():
    run_test_harness()

# run the test harness for evaluating a model
def run_test_harness():
    # load dataset
    (trainX, trainY), (validX, validY), (testX, testY) = load_dataset()
    
    # define model
    # Choose between original, dropout, weight decay or data-augmentation model (uncomment desired)
    # model = define_model() # Basic model
    model = dropout_model() # With dropout
    
    # fit model, should eventually get validation data from training data
    history = model.fit(trainX, trainY, epochs=50, batch_size=64, validation_data=(validX, validY), verbose=1)
    # evaluate model
    _, acc = model.evaluate(testX, testY, verbose=1)
    print('> %.3f' % (acc * 100.0))
    # learning curves
    # summarize_diagnostics(history, "test_plot")

def define_model():
    """
    Define basic model.
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def dropout_model():
    """
    Define model with dropout.
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    # Add dropout after the first convolutional layer
    model.add(Dropout(0.20))  # We set dropout to %20 &retain %80 of nodes (can be adjusted)
    
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    # Add dropout after the second set of convolutional layers
    model.add(Dropout(0.20))  
    
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    # Add dropout after the third set of convolutional layers
    model.add(Dropout(0.20))  
    
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    # Add dropout before the fully connected layer
    model.add(Dropout(0.2))  # Adjust dropout rate as needed
    
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

"""
# plot diagnostic learning curves
def summarize_diagnostics(history, filename):
    pyplot.figure(figsize=(7, 8))
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='validation')
    pyplot.legend()
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='validation')
    # save plot to file
    pyplot.legend()
    pyplot.savefig(filename + ".png")
    pyplot.close()
"""

#load cifar10 dataset into a a train and test set
def load_dataset(valid_size=5000):
    (trainX, trainy), (testX, testy) = cifar10.load_data()

    trainY = to_categorical(trainy)
    testY = to_categorical(testy)

    trainX = normalize_data(trainX)
    testX = normalize_data(testX)

    train_size = trainX.shape[0] - valid_size

    validX = trainX[train_size:train_size + valid_size]
    validY = trainY[train_size:train_size + valid_size]

    trainX = trainX[:train_size]
    trainY = trainY[:train_size]

    return (trainX, trainY), (validX, validY), (testX, testY)

#converts image data from unsigned ints in 0-255 to floats i 0-1 range
def normalize_data(X):
    X_norm = X.astype('float32')
    X_norm = X_norm / 255.0
    return X_norm

if __name__ == "__main__":
    main()