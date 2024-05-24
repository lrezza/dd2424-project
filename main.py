import numpy as np
import math
import time
import scipy
from matplotlib import pyplot
from keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout  # Import Dropout layer
from tensorflow.keras.optimizers import SGD, Adam, AdamW
from keras.regularizers import l2
from keras.datasets import cifar10
# from keras.preprocessing.image import ImageDataGenerator
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.optimizers.schedules import CosineDecay, PiecewiseConstantDecay, CosineDecayRestarts

def main():
    # Choose which test harness to run
    time_init = time.time()
    #run_test_harness() # Uncomment for original, dropout or weight decay model 
    #run_augmentation_harness() # Uncomment for data augmentation model
    #time_init = time.time()
    run_experimentation()
    time_finish = time.time()

    print(f"Everything took {time_finish - time_init} seconds.")

def learning_rate_scheduler(epoch, learning_rate):
    """
    Implementation based on: https://keras.io/api/callbacks/learning_rate_scheduler/'
    """

    if epoch < 10:
        return learning_rate
    else:
        return learning_rate * math.exp(-0.1)
    



# run the test harness for evaluating a basic, dropout or weight decay model
def run_test_harness():
    # load dataset
    (trainX, trainY), (validX, validY), (testX, testY) = load_dataset()
    
    # define model
    # Choose between basic, dropout or weight decay model (uncomment desired)
    # model = define_model() # Basic model
    # model = dropout_model() # With dropout
    model = dropout_model() # With weight decay
    
    # fit model, should eventually get validation data from training data
    history = model.fit(trainX, trainY, epochs=100, batch_size=64, validation_data=(validX, validY), verbose=1)
    # evaluate model
    _, acc = model.evaluate(testX, testY, verbose=1)
    print('> %.3f' % (acc * 100.0))
    # learning curves
    summarize_diagnostics(history, "test")

# run the test harness for evaluating a data augmentation model
def run_augmentation_harness():
    """
    Updated test harness to support data augmentation.
    """
    # load dataset
    (trainX, trainY), (validX, validY), (testX, testY) = load_dataset()

    # define model
    #model = define_model() # Basic model
    model = combined_model() #combined model with dropout and batch normalization
    batch_size = 64
    # create data generator
    datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    # prepare iterator
    it_train = datagen.flow(trainX, trainY, batch_size=batch_size)
    
    # fit model
    steps = math.ceil(trainX.shape[0] / batch_size)

    print(steps)
    print(trainX.shape)     
    
    #epochs = 100 #for most models
    epochs = 400 #for combined model

    history = model.fit(it_train, steps_per_epoch=steps, epochs=epochs, 
                        validation_data=(validX, validY), verbose=1)
    
    # evaluate model
    _, acc = model.evaluate(testX, testY, verbose=1)
    print('> %.3f' % (acc * 100.0))
    # learning curves
    summarize_diagnostics(history, "test")

def run_experimentation():
    """
    Run the various experimentation suggestions.

    The first three tests uses the combined_model() with selectable optimizer.
    Only ordering_complementary_test() uses the variable_model().

    """
    epochs = 50 # TODO: Final run with 100 or 400
    # (trainX, trainY), (validX, validY), (testX, testY) = load_dataset(zero_mean=False)
    #data = load_dataset(zero_mean=False)    # Combined for easier pass to run_and_evaluate()
    data = None

    ## UTILS
    def run_and_evaluate(model_type, data, optimizer, filename, suffix, order="", filter="", learning_rate=0.001, zero_mean=False):
        (trainX, trainY), (validX, validY), (testX, testY) = load_dataset(zero_mean=zero_mean)
        print("Shape check: ", len(trainX), len(validX))

        if model_type == 'variable':
            model = variable_model(optimizer=optimizer, order=order, filter=filter)
        else:
            model = combined_model(optimizer=optimizer, learning_rate=learning_rate)
        batch_size = 64

        datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
        it_train = datagen.flow(trainX, trainY, batch_size=batch_size)

        steps = math.ceil(trainX.shape[0] / batch_size)
        #steps = math.floor(trainX.shape[0] / batch_size)

        print("Number of steps:", steps, "")
        print("TrainX shape:", trainX.shape)

        history = model.fit(it_train, steps_per_epoch=steps, epochs=epochs, validation_data=(validX, validY), verbose=1)

        _, acc = model.evaluate(testX, testY, verbose=1)
        print('ACCURACY:')
        print('> %.3f' % (acc * 100.0))
        print("\n")

        summarize_diagnostics(history, filename=filename, suffix=suffix)


    ## TESTS
    def zero_mean_test(epochs=epochs):
        epochs=50
        #data = load_dataset(zero_mean=True)
        
        run_and_evaluate('combined', data, optimizer='SGD', filename='zero_mean', suffix=' for Zero-Mean test')

        print("\n -- Zero Mean experiment done --\n")
    

    def optimizer_variation_test(epochs=epochs):
        # (trainX, trainY), (validX, validY), (testX, testY) = load_dataset(zero_mean=False)
        # data = load_dataset(zero_mean=False)    # Combied for easier pass to run_and_evaluate()

        optimizers = ['Adam', 'AdamW']

        for optimizer in optimizers:
            run_and_evaluate('combined', data, optimizer=optimizer, filename=optimizer, suffix=f' for optimizer {optimizer}')
            
        print("\n -- Optimizer experiment done --\n")
    

    def learning_rate_scheduler_test():
        # Warm-up: CosineDecay - use warmup target and steps
        # Step Decay: ExponentialDecay - use Staircase argument
        # w. Restarts: CosineDecayRestarts - use t_mul and m_mul

        # Cosine Annealing with Warm-Up
        warmup_decay_lr = CosineDecay(initial_learning_rate=1e-6, decay_steps=10, alpha=0.1, warmup_target=1e-3, warmup_steps=10)
        
        # Step Decay using 
        boundaries = [25, 50, 75]
        values = [0.1, 0.01, 0.001, 0.0001]
        step_decay_lr = PiecewiseConstantDecay(boundaries=boundaries, values=values)

        # Cosine Annealing with restarts
        cosine_ann_lr = CosineDecayRestarts(initial_learning_rate=1e-6, first_decay_steps=10, t_mul=2.0, m_mul=1.0, alpha=0.0001)

        run_and_evaluate('variable', data, optimizer='SGD', filename='warm-up', suffix='for LR Warm-Up + Cosine', learning_rate=warmup_decay_lr)
        run_and_evaluate('variable', data, optimizer='SGD', filename='step-decay', suffix='for LR Step Decay', learning_rate=step_decay_lr)
        run_and_evaluate('variable', data, optimizer='SGD', filename='cosine-restart', suffix='for LR Cosine w. Restarts', learning_rate=cosine_ann_lr)

        print("\n -- Optimizer experiment done --\n")
    

    def ordering_complementary_test():
        orders = ['pre', 'post']
        filters = ['BatchNorm', 'Dropout', 'Both']

        #(X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test) = load_dataset(zero_mean=False)

        # This test simply changes the order of BatchNorm and Dropout where applicable.
        print("\nOrdering Test\n")
        run_and_evaluate('variable', data, optimizer='SGD', filename='post_order_test', suffix='for reverse order', order='post', filter=filters[2])


        # This uses the default order (pre) but filters out either BatchNorm or Dropout
        print("\nFiltering test\n")
        for filter in filters:
            order = 'pre'
            # model = variable_model(optimizer='SDG', order='post', filter=filter)
            
            filename = f'{order}_{filter}'
            run_and_evaluate('variable', data, optimizer='SGD', filename=filename, suffix=f' | order: {order} filter: {filter}', order=order, filter=filter)
        
        print("\n -- Optimizer experiment done --\n")
    
    
    # Run tests
    zero_mean_test()
    #optimizer_variation_test()
    #learning_rate_scheduler_test()
    #ordering_complementary_test()

    print("All tests complete.")

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

""" def dropout_model():

    #Define model with dropout.
    
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
    return model """

def dropout_model():
    #Define model with dropout.
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    # Add dropout after the first set of convolutional layers
    model.add(Dropout(0.20))  # We set dropout to %20 &retain %80 of nodes (can be adjusted)
    
    
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    # Add dropout after the second set of convolutional layers
    model.add(Dropout(0.20))  
    
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    # Add dropout after the third set of convolutional layers
    model.add(Dropout(0.20))  

    model.add(Flatten())
    
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    # Add dropout before the fully connected layer
    model.add(Dropout(0.2))  # Adjust dropout rate as needed
    model.add(Dense(10, activation='softmax'))

    # compile model
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def decay_model():
    """
    Define model with weight decay.
    """
    model = Sequential()
    weight_decay = 0.001 # Set weight decay to standard value (can be adjusted)
    # First convolutional block with ReLU activation, He uniform initialization,
    # same padding to maintain spatial dimensions, and L2 weight decay for regularization
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform',
                   padding='same', kernel_regularizer=l2(weight_decay), input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform',
                   padding='same', kernel_regularizer=l2(weight_decay)))
    model.add(MaxPooling2D((2, 2)))
    
    # Second convolutional block with the same configuration as the first block
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform',
                   padding='same', kernel_regularizer=l2(weight_decay)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform',
                   padding='same', kernel_regularizer=l2(weight_decay)))
    model.add(MaxPooling2D((2, 2)))
    
    # Third convolutional block with the same configuration
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform',
                   padding='same', kernel_regularizer=l2(weight_decay)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform',
                   padding='same', kernel_regularizer=l2(weight_decay)))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Flatten())
    
    # Dense layer with ReLU activation, He uniform initialization, and L2 weight decay
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(weight_decay)))
    
    model.add(Dense(10, activation='softmax'))
    
    # Compile model with SGD optimizer, categorical crossentropy loss, and accuracy metric
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def combined_model(optimizer: str='SGD', learning_rate=0.001):
    # Param: optimizer = ['SDG', 'Adam', 'AdamW']
    print("ENTERED COMBINED")
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    # compile model
    # opt = SGD(lr=0.001, momentum=0.9) # lr didn't work for me // Alex

    opt = SGD(learning_rate=learning_rate, momentum=0.9)
    if optimizer == 'Adam':
        opt = Adam(learning_rate=0.001) # NOTE: Interpreted instructions to not include momentum
    elif optimizer == 'AdamW':
        opt = AdamW(learning_rate=0.001)
    #print("Opt type: ", type(opt))
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def variable_model(optimizer: str='SGD', order: str='pre', filter: str='drop'):
    """
    Variable model based on the combined model.

    :param order: pre/post - post changes default order (post only for clarity, evaluated as 'else')
    :param filter: BatchNorm/DropOut/Both - NOTE: filter is what to ADD, not what to filter away
    """

    def add_filtered(model, order, filter, exception: bool=False):
        """
        Adds only BatchNorm, Droputout or both based on the filter parameter.
        :param exception: Used by the input layer to avoid adding BatchNorm instad of nothing when using 'post' order
        :return: model
        """
        if not filter == 'Dropout':
            if not exception:   # See function doc
                model.add(BatchNormalization) if order == 'Pre' else model.add(Dropout(0.2))

        if not filter == 'BatchNorm':
            model.add(Dropout(0.2)) if order == 'Pre' else model.add(BatchNormalization())
        
        return model
    
    print(f"\nCurrent model has order: {order} and fiter: {filter}\n")  # Debugging
    print(order, filter)
    print(type(order), type(filter))

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    #if not filter == 'Dropout': model.add(BatchNormalization())
    model = add_filtered(model, order, filter, exception=True)
    
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    #if not filter == 'Dropout': model.add(BatchNormalization())
    model = add_filtered(model, order, filter)
    model.add(MaxPooling2D((2, 2)))
    #if not filter == 'BatchNorm': model.add(Dropout(0.2))
    model = add_filtered(model, order, filter)
    
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    #if not filter == 'Dropout': model.add(BatchNormalization())
    model = add_filtered(model, order, filter)

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    #if not filter == 'Dropout': model.add(BatchNormalization())
    model = add_filtered(model, order, filter)
    model.add(MaxPooling2D((2, 2)))
    #if not filter == 'BatchNorm': model.add(Dropout(0.3))
    model = add_filtered(model, order, filter)
    
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    #if not filter == 'Dropout': model.add(BatchNormalization())
    model = add_filtered(model, order, filter)

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    #if not filter == 'Dropout': model.add(BatchNormalization())
    model = add_filtered(model, order, filter)
    model.add(MaxPooling2D((2, 2)))
    #if not filter == 'BatchNorm': model.add(Dropout(0.4))
    model = add_filtered(model, order, filter)

    model.add(Flatten())
    
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    #if not filter == 'Dropout': model.add(BatchNormalization())
    #if not filter == 'BatchNorm': model.add(Dropout(0.5))
    model = add_filtered(model, order, filter)
    model = add_filtered(model, order, filter)
    model.add(Dense(10, activation='softmax'))

    if optimizer == 'SGD':
        opt = SGD(learning_rate=0.001, momentum=0.9)
    elif optimizer == 'Adam':
        opt = Adam(learning_rate=0.001) # NOTE: Interpreted instructions to not include momentum
    elif optimizer == 'AdamW':
        opt = AdamW(learning_rate=0.001)
    print("Opt type: ", type(opt))
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # Debugging checking correct construction
    print("\n ###### MODEL SUMMARY #######\n")
    model.summary()
    print("\n\n")

    return model


# plot diagnostic learning curves
def summarize_diagnostics(history, filename, suffix: str=""):
    pyplot.figure(figsize=(7, 8))
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss' + suffix)
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='validation')
    pyplot.legend()
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy' + suffix)
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='validation')
    # save plot to file
    pyplot.legend()
    pyplot.savefig(filename + ".png")
    pyplot.close()

#load cifar10 dataset into a a train and test set
def load_dataset(valid_size=5000, zero_mean: bool=False):
    (trainX, trainy), (testX, testy) = cifar10.load_data()

    trainY = to_categorical(trainy)
    testY = to_categorical(testy)

    trainX = normalize_data(trainX, zero_mean=zero_mean)
    testX = normalize_data(testX, zero_mean=zero_mean)

    train_size = trainX.shape[0] - valid_size

    validX = trainX[train_size:train_size + valid_size]
    validY = trainY[train_size:train_size + valid_size]

    trainX = trainX[:train_size]
    trainY = trainY[:train_size]

    return (trainX, trainY), (validX, validY), (testX, testY)

#converts image data from unsigned ints in 0-255 to floats i 0-1 range
def normalize_data(X, zero_mean: bool=False):
    X_norm = X.astype('float32')
    X_norm /= 255.0
    #zero_mean=False
    if zero_mean:
        X_norm -= X_norm.mean()
        X_norm /= X_norm.std()
    return X_norm

if __name__ == "__main__":
    main()