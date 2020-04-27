import os
import glob
import keras
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras.models import load_model
import numpy as np

# # example of loading an image with the Keras API
# from keras.preprocessing.image import load_img
# # load the image
# img = load_img('C:/Users/Mikian/Desktop/processimgs/jacobtest.PNG')
# # report details about the image
# print(type(img))
# print(img.format)
# print(img.mode)
# print(img.size)
# # show the image
# img.show()

imgPath = 'C:/Users/Mikian/Desktop/processimgs/'

Height = 224
Width  = 224
BatchSize = 1
Version = 1

fullPath = 'D:/Kaggle-Autism/experiments/Autism/20200411-162232/weights/weights-improvement-38-0.9000.hdf5'
fullPath = 'D:/Kaggle-Autism/experiments/Autism/20200411-172305/weights/weights-improvement-05-0.8500.hdf5'

print("Loading:", fullPath)
model = load_model(fullPath)

def preprocess_input_new(x):
    img = preprocess_input(keras.preprocessing.image.img_to_array(x), version = 2)
    return np.array(keras.preprocessing.image.array_to_img(img))


predictGen = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_input_new).flow_from_directory(
        imgPath,
        target_size=(Height, Width),
        batch_size=BatchSize,
        shuffle=False)

results = model.predict_generator(predictGen, verbose=0)
print(results)
