from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D,Flatten, Activation, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore')

def main():
    BS = 40
    # load images #
    #-------------#

    train_path = 'train'
    valid_path = 'valid'
    test_path  = 'test'

    train_batches = ImageDataGenerator(rescale=1./255)
    train_batches = train_batches.flow_from_directory(directory=train_path,
                                                      target_size=(32,32),
                                                      classes=['car', 'plane', 'ship', 'truck'],
                                                      batch_size=BS)

    valid_batches = ImageDataGenerator(rescale=1./255)
    valid_batches = valid_batches.flow_from_directory(directory=valid_path,
                                                      target_size=(32,32),
                                                      classes=['car', 'plane', 'ship', 'truck'],
                                                      batch_size=BS)

    test_batches = ImageDataGenerator(rescale=1./255)
    test_batches = test_batches.flow_from_directory(directory=test_path,
                                                    target_size=(32,32),
                                                    classes=['car', 'plane', 'ship', 'truck'],
                                                    batch_size=BS)


    # calculate steps #
    #-----------------#

    STEPS_PER_EPOCH =  round( len(train_batches.filenames)/ BS)
    VALIDATION_STEP =  round( len(valid_batches.filenames) / BS)



    # building and training cnn #
    # ---------------------------#

    model = Sequential()

    model.add(Conv2D(32, (3,3), input_shape=(32,32,3),padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3),padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128, (3,3)))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'] )

    history = model.fit_generator(generator=train_batches,
                        steps_per_epoch=STEPS_PER_EPOCH,
                        validation_data=valid_batches,
                        validation_steps=VALIDATION_STEP,
                        epochs=13,
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
    plt.plot(epochs, val_loss, 'g-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # plot accuracy #
    #---------------#
    plt.figure()
    plt.plot(epochs, train_acc, 'ro', label='Training accuracy')
    plt.plot(epochs, val_acc, 'g-', label='Validation accuracy')
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
            if n >= len(image_batch):
                break
            plt.subplot(5, 5, n + 1)
            plt.imshow(image_batch[n])
            plt.title(label_batch[n])
            plt.axis('off')
        plt.show()

    # uncomment to plot all test images #
    #-----------------------------------#
    # for _ in test_batches:
    #     image_batch,label_batch = _
    #     show_batch(image_batch, label_batch)

    # uncomment to plot one batch #
    #-----------------------------#
    # image_batch,label_batch = next(test_batches)
    # show_batch(image_batch, label_batch)




if __name__ == '__main__':
    main()