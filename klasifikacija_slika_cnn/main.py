import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D,Flatten, Activation, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator



def load():
    train_path = 'train'
    valid_path = 'valid'
    test_path  = 'test'

    global train_batches, valid_batches, test_baches

    train_batches = ImageDataGenerator()\
        .flow_from_directory(directory=train_path,
                             target_size=(64,64),
                             classes=['dog', 'cat'],
                             batch_size=BS)

    valid_batches = ImageDataGenerator()\
        .flow_from_directory(directory=valid_path,
                             target_size=(64,64),
                             classes=['dog', 'cat'],
                             batch_size=BS)
    test_baches = ImageDataGenerator()\
        .flow_from_directory(directory=test_path,
                             target_size=(64,64),
                             classes=['dog', 'cat'],
                             batch_size=BS)

    global  STEPS_PER_EPOCH, VALIDATION_STEP
    STEPS_PER_EPOCH =  round( len(train_batches.filenames)/ BS)
    VALIDATION_STEP =  round( len(valid_batches.filenames) / BS)


def train():
    # building and training cnn
    model = Sequential()
    model.add(Conv2D(64, (3,3), input_shape=(64,64,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3,3), input_shape=(64,64,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3,3), input_shape=(64,64,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3,3), input_shape=(64,64,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3,3), input_shape=(64,64,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3,3), input_shape=(64,64,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))


    #classifier
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'] )

    #train
    model.fit_generator(generator=train_batches,
                        steps_per_epoch=STEPS_PER_EPOCH,
                        validation_data=valid_batches,
                        validation_steps=VALIDATION_STEP,
                        epochs=5,
                        verbose=2)


    predictions = model.predict_generator(test_baches,steps=1, verbose=0)
    print(model.summary())


if __name__ == '__main__':
    train_batches = valid_batches = test_baches = None
    BS = 50
    STEPS_PER_EPOCH = VALIDATION_STEP = None

    #load images for model
    load()
    #set steps for training
    STEPS_PER_EPOCH =  round( len(train_batches.filenames)/ BS)
    VALIDATION_STEP =  round( len(valid_batches.filenames) / BS)
    #train nn
    train()