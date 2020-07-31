from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import numpy as np

seed = 7
np.random.seed(seed)
num_classes = 10 + 26
batch_size = 16

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.15)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory='./samples',
    target_size=(227, 227),
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

valid_generator = train_datagen.flow_from_directory(
    directory='./samples',
    target_size=(227, 227),
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=True,
    seed=42
)

test_generator = train_datagen.flow_from_directory(
    directory='./output',
    target_size=(227, 227),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    shuffle=False,
    seed=42
)

model = load_model('./model.hdf5')

loss, accuracy = model.evaluate(test_generator)
print(f'loss: {loss}, accuracy: {accuracy}')
