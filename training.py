# Import dependency 
import os 
import cv2
import keras
import numpy as np
from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from IPython.display import Image
from keras.optimizers import Adam




IMAGE_DATA = '/u01/duonglt/tf_image_recognition/detect_face/data'

path_dir = '/u01/duonglt/tf_image_recognition/detect_face/'
model_file = os.path.join(path_dir, 'checkpoint.hdf5')
model_file_final_json = os.path.join(path_dir, 'model_file_final_json.json')
model_file_final = os.path.join(path_dir, 'model_file_final.h5')

purpose = 'faces_me'
epochs = 100


IMAGE_SIZE = 224

images = []
labels = []


#========== Process data

def prepare_image(file):
    img_path = ''
    img = image.load_img(img_path + file, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

# Generate dataset

for path, subdirs, files in os.walk(IMAGE_DATA):
    for name in files:
        img_path = os.path.join(path, name)
        if (os.stat(img_path).st_size != 0):
            images.append(prepare_image(img_path))
            labels.append(path.split('/')[-1])

images = np.array(images)
images = np.squeeze(images, axis=1)


#========== Convert target feature
mapped_labels = list(map(lambda x: 1 if x == purpose else 0, labels))

from keras.utils import np_utils
y_data = np_utils.to_categorical(mapped_labels)


#========== Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(images, y_data, test_size=0.2)


#========== Init model
model = keras.applications.mobilenet.MobileNet(classes=2, weights=None)
model.summary()


#========== Conffig model
model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint


checkpoint = ModelCheckpoint(model_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


#========== Run training
model.fit(x=X_train, y=y_train, batch_size=16, epochs=epochs, verbose=1, validation_data=(X_test, y_test), callbacks=callbacks_list)


#========== Save model
# serialize model to JSON
model_json = model.to_json()


with open(model_file_final_json, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(model_file_final)
print("Saved model to disk")