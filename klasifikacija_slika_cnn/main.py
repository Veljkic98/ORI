import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D,Flatten, Activation, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore')
def main():

    # load images #
    #-------------#

    train_path = 'train'
    valid_path = 'valid'
    test_path  = 'test'

    train_batches = ImageDataGenerator(rescale=1./255).flow_from_directory(directory=train_path,
                                                                           target_size=(32,32),
                                                                           classes=['car', 'plane', 'ship'],
                                                                           batch_size=BS)

    valid_batches = ImageDataGenerator(rescale=1./255).flow_from_directory(directory=valid_path,
                                                                           target_size=(32,32),
                                                                           classes=['car', 'plane', 'ship'],
                                                                           batch_size=BS)

    test_batches = ImageDataGenerator(rescale=1./255).flow_from_directory(directory=test_path,
                                                                         target_size=(32,32),
                                                                         classes=['car', 'plane', 'ship'],
                                                                         batch_size=BS)


    # calculate steps #
    #-----------------#

    STEPS_PER_EPOCH =  round( len(train_batches.filenames)/ BS)
    VALIDATION_STEP =  round( len(valid_batches.filenames) / BS)



    # building and training cnn #
    # ---------------------------#

    model = Sequential()

    model.add(Conv2D(32, (3,3), input_shape=(32,32,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'] )

    history = model.fit_generator(generator=train_batches,
                        steps_per_epoch=STEPS_PER_EPOCH,
                        validation_data=valid_batches,
                        validation_steps=VALIDATION_STEP,
                        epochs=10,
                        verbose=2)



    # history data to plot#
    #---------------------#

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(val_loss) + 1)

    # plot loss #
    #-----------#
    plt.figure()
    plt.plot(epochs, train_loss, 'ro', label='Training loss')
    plt.plot(epochs, val_loss, 'b-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # plot accuracy #
    #---------------#
    plt.figure()
    plt.plot(epochs, train_acc, 'ro', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b-', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()



    # testing #
    #---------#
    # predictions = model.predict(x=test_batches, verbose=0)
    # print(predictions)
    # print(numpy.round(predictions))

    # plot testing images #
    #---------------------#
    def show_batch(image_batch, label_batch):
        plt.figure(figsize=(10, 10))
        for n in range(25):
            plt.subplot(5, 5, n + 1)
            plt.imshow(image_batch[n])
            plt.title(label_batch[n])
            plt.axis('off')
        plt.show()

    # uncomment to plot all test images #
    #-----------------------------------#
    for _ in test_batches:
        image_batch,label_batch = _
        show_batch(image_batch, label_batch)

    # uncomment to plot one batch #
    #-----------------------------#
    # image_batch,label_batch = next(test_batches)
    # show_batch(image_batch, label_batch)




if __name__ == '__main__':
    BS = 40
    # train_batches = valid_batches = test_batches = None
    # STEPS_PER_EPOCH = VALIDATION_STEP = None
    main()