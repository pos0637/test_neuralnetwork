import keras
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np

seed = 7
np.random.seed(seed)
num_classes = 10 + 26
batch_size = 64

train_datagen = ImageDataGenerator(
    rescale=1./255,
    featurewise_center=True,
    zoom_range=0.3)
valid_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.3, featurewise_center=True)
test_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.15,    
    featurewise_center=True
)

train_generator = train_datagen.flow_from_directory(
    directory='./samples',
    target_size=(227, 227),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

valid_generator = valid_datagen.flow_from_directory(
    directory='./samples',
    target_size=(227, 227),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=True,
    seed=42
)

test_generator = test_datagen.flow_from_directory(
    directory='./output',
    target_size=(227, 227),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=True,
    seed=42
)

optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)

model = keras.Sequential()
model.add(Conv2D(96, (11, 11), strides=(4, 4), input_shape=(227, 227, 1),
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

model.compile(optimizer=optimizer, loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

checkpoint = ModelCheckpoint(
    './weights.hdf5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max')
tensorBoard = TensorBoard(
    batch_size=batch_size
)

model.fit(
    train_generator,
    validation_data=test_generator,
    steps_per_epoch=train_generator.n // train_generator.batch_size,
    epochs=50,
    callbacks=[checkpoint, tensorBoard])

model.load_weights('./weights.hdf5')
model.save('model.hdf5')

loss, accuracy = model.evaluate(valid_generator)
print(f'loss: {loss}, accuracy: {accuracy}')
