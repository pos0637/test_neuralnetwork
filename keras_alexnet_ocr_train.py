import keras
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np

seed = 7
np.random.seed(seed)
num_classes = 10 + 26
batch_size = 16

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.15)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory='./samples',
    target_size=(120, 200),
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

valid_generator = train_datagen.flow_from_directory(
    directory='./samples',
    target_size=(120, 200),
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=True,
    seed=42
)

test_generator = train_datagen.flow_from_directory(
    directory='./output',
    target_size=(120, 200),
    color_mode='rgb',
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=42
)

model = keras.Sequential()
model.add(Conv2D(96, (11, 11), strides=(4, 4), input_shape=(120, 200, 3),
                 padding='valid', activation='relu', kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same',
                 activation='relu', kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same',
                 activation='relu', kernel_initializer='uniform'))
model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same',
                 activation='relu', kernel_initializer='uniform'))
model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same',
                 activation='relu', kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='sgd', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

checkpoint = ModelCheckpoint(
    './weights.hdf5',
    monitor='accuracy',
    save_best_only=True,
    mode='max')
tensorBoard = TensorBoard(
    batch_size = batch_size
)

model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // train_generator.batch_size,
    epochs=50,
    callbacks=[checkpoint, tensorBoard])

model.load_weights('./weights.hdf5')
model.save('model.hdf5')

loss, accuracy = model.evaluate(valid_generator)
print(f'loss: {loss}, accuracy: {accuracy}')
