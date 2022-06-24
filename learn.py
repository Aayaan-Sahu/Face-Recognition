import os
import json
import numpy as np
from imutils import paths
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Google Colab
# from google.colab import drive
# drive.mount('/content/gdrive', force_remount=True)
# data_dir = '/content/gdrive/My Drive/COLABS/face_detector/data' # the stuff after gdrive/ should be the directory to your data within Google Drive

# Local
data_dir = './data'

labels = []
for folder in os.listdir(data_dir):
    labels.append(folder)

IMAGE_DIMENSION = 224
image_paths = list(paths.list_images(data_dir))

data = []
labels = []
for image_path in image_paths:
    label = image_path.split('/')[-1].split('_')[0]

    image = load_img(image_path, target_size=(IMAGE_DIMENSION, IMAGE_DIMENSION))
    image = img_to_array(image)
    image = preprocess_input(image)

    data.append(image)
    labels.append(label)
data = np.array(data, dtype='float32')
labels = np.array(labels)

label_encoder = LabelBinarizer()
labels = to_categorical(label_encoder.fit_transform(labels))

(train_X, test_X, train_y, test_y) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

augmentation_generator = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip = True,
    fill_mode='nearest',
)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

head_model = base_model.output
head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
head_model = Flatten(name='flatten')(head_model)
head_model = Dense(units=128, activation='relu')(head_model)
head_model = Dropout(rate=0.5)(head_model)
num_classes = 0
for base, dirs, files in os.walk(data_dir):
    for directory in dirs:
        num_classes += 1
print(num_classes)
head_model = Dense(units=num_classes, activation='softmax')(head_model)

model = Model(inputs=base_model.input, outputs=head_model)

for layer in base_model.layers:
  layer.trainable = False


INIT_LR = 1e-4
EPOCHS = 20
print('[INFO] compiling model...')
optimizer = Adam(learning_rate=INIT_LR, decay=(INIT_LR / EPOCHS))
model.compile(
    loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

BATCH_SIZE = 32
print('[INFO] training head...')
H = model.fit(
    augmentation_generator.flow(x=train_X, y=train_y, batch_size=BATCH_SIZE),
    steps_per_epoch=(len(train_X) // BATCH_SIZE),
    validation_data=(test_X, test_y),
    validation_steps=(len(test_X) // BATCH_SIZE),
    epochs=EPOCHS
)

print('[INFO] saving model...')
model.save(filepath='detector', save_format='h5')

indices = list(np.unique(labels))
corr_labels = list(label_encoder.inverse_transform(np.unique(labels)))
label_dict = dict()

for i, l in zip(indices, corr_labels):
  label_dict[int(i)] = l

with open("labels.json", "w") as outfile:
  json.dump(label_dict, outfile)