from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=1024)
seed = 7
np.random.seed(seed)
num_classes = 10 + 26
batch_size = 64

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.15)
test_datagen = ImageDataGenerator(
    rescale=1./255,
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

valid_generator = train_datagen.flow_from_directory(
    directory='./samples',
    target_size=(227, 227),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=True,
    seed=42
)

test_generator = train_datagen.flow_from_directory(
    directory='./output',
    target_size=(227, 227),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical',
    shuffle=False,
    seed=42
)

model = load_model('./model.hdf5')

loss, accuracy = model.evaluate(test_generator)
print(f'loss: {loss}, accuracy: {accuracy}')

pred = model.predict(test_generator)
pred = np.argmax(pred, axis=1)
mat = confusion_matrix(test_generator.classes, pred)
print(mat)
# 1print('Classification Report')
# target_names = ['Cats', 'Dogs', 'Horse']
# print(classification_report(validation_generator.classes, y_pred, target_names=target_names))

# mat = mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]
# mat = np.around(mat, decimals=2)
plt.figure(figsize=(8, 8))
sns.heatmap(mat, annot=True, cmap='Blues')
plt.ylim(0, 10)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()
