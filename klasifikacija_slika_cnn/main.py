import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D,Flatten
from keras.preprocessing.image import ImageDataGenerator
train_path = 'train'
valid_path = 'valid'
test_path  = 'test'

train_batches = ImageDataGenerator()\
    .flow_from_directory(directory=train_path,
                         target_size=(64,64),
                         classes=['dog', 'cat'],
                         batch_size=10)

valid_batches = ImageDataGenerator()\
    .flow_from_directory(directory=valid_path,
                         target_size=(64,64),
                         classes=['dog', 'cat'],
                         batch_size=10)
test_baches = ImageDataGenerator()\
    .flow_from_directory(directory=test_path,
                         target_size=(64,64),
                         classes=['dog', 'cat'],
                         batch_size=10)

# building and training cnn
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))

model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'] )

#train
model.fit_generator(generator=train_batches,
                    steps_per_epoch=4,
                    validation_data=valid_batches,
                    validation_steps=4,
                    epochs=5,
                    verbose=1)