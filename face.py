import os
import glob
import time
import keras
from PIL import Image
from os import listdir
from shutil import copyfile
from os.path import isfile, join
from matplotlib import pyplot as plt
from keras_vggface.vggface import VGGFace
from keras.engine import Input
from keras import applications
from keras.models import Model
import tensorflow as tf
import numpy as np
from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from keras_vggface.utils import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
from MLEXPS.MLEXPS import *
from keras import backend as K
import random

random.seed(42)
tf.random.set_seed(42)

# Providing more training examples within certain distributions of age, gender, and race will increase the model's accuracy.

Height = 224
Width  = 224
BatchSize = 24
lr_rate=.0015
Version = 5
load_model = False
model_path = ''
accuracy = 0
accuracyCount = 0
trainableCount = 30

def SaveModelImage(Model, Title):
    keras.utils.vis_utils.plot_model(Model, to_file=Title, show_shapes=True, show_layer_names=True)
    return

def Summary(Model):
    print(Model.summary())
    return

def resnet():
    BaseModel = applications.resnet50.ResNet50(weights= None, include_top=False, input_shape= (224,224,3))
    last_layer = BaseModel.get_layer('activation_49').output
    print('here')
    return model

def MakeModel(dlsize):
    BaseModel = VGGFace(model='senet50', include_top=False, input_shape=(Height, Width, 3), pooling='avg')
    last_layer = BaseModel.get_layer('avg_pool').output

    x = keras.layers.Flatten(name='flatten')(last_layer)

    x = keras.layers.Dense(128, kernel_regularizer = keras.regularizers.l2(l = 0.015), activation='relu')(x)
    x = keras.layers.Dropout(rate=.4, seed=42)(x)

    out = keras.layers.Dense(2, activation='softmax', name='classifier')(x)
    DerivedModel = keras.Model(BaseModel.input, out)

    # # Everything is trainingable
    # # Weights are used at init
    # for layer in DerivedModel.layers:
    #     layer.trainable = True
    #
    #
    # # Everything in the base model is frozen
    # # Only top layers are trainable
    # for layer in BaseModel.layers:
    #     layer.trainable = False

    # base
    for layer in DerivedModel.layers:
        layer.trainable = False
    for layer in DerivedModel.layers[-trainableCount:]:
        layer.trainable = True


    DerivedModel.compile(keras.optimizers.Adam(lr=lr_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return DerivedModel

def clearWeights(model):
    weights = model.get_weights()
    for weight in weights:
        weight = K.zeros(weight.shape, dtype=np.float64)
    model.set_weights(weights)
    return model

def preprocess_input_new(x):
    img = preprocess_input(keras.preprocessing.image.img_to_array(x), version = 2)
    return keras.preprocessing.image.array_to_img(img)

class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):

  def __init__(self, trainableCount=30):
    print('working')
    super(EarlyStoppingAtMinLoss, self).__init__()
    self.epochCount = []
    self.trainableCount = trainableCount
    self.max = 0

  def on_train_begin(self, logs=None):
    self.accuracyCount = 0
    self.accuracy = 0

  def on_epoch_end(self, epoch, logs=None):
    self.max = len(self.model.layers)
    print("Ending Epoch")
    if logs['val_accuracy'] > self.accuracy:
        self.accuracy = logs['val_accuracy']
        self.accuracyCount = 0
    else:
        self.accuracyCount+=1

    if self.accuracyCount >= 10 * (len(self.epochCount)+1):
        self.epochCount.append(epoch)
        print('Adding train layers')
        self.accuracyCount = 0
        self.trainableCount += 10
        if self.trainableCount >= self.max:
            self.trainableCount = self.max
        for layer in self.model.layers:
            layer.trainable = False
        for layer in self.model.layers[-self.trainableCount:]:
            layer.trainable = True
        self.model.compile(keras.optimizers.Adam(lr=lr_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    print(self.epochCount)

if __name__ == "__main__":
    timestr = time.strftime("%Y%m%d-%H%M%S")
    model = MakeModel(1024)
    # model = resnet()
    # model = clearWeights(model)
    model.compile(keras.optimizers.Adam(lr=lr_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    TrainPath = 'D:/Autism-Data/Kaggle/v' + str(Version) + '/train'
    ValidPath = 'D:/Autism-Data/Kaggle/v' + str(Version) + '/valid'
    TestPath  = 'D:/Autism-Data/Kaggle/v' + str(Version) + '/test'

    TrainGen = keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=preprocess_input_new,
            horizontal_flip=True,
            rotation_range=45,
            width_shift_range=.01,
            height_shift_range=.01).flow_from_directory(
            TrainPath,
            target_size=(Height, Width),
            batch_size=BatchSize)

    ValidGen = keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=preprocess_input_new).flow_from_directory(
            ValidPath,
            target_size=(Height, Width),
            batch_size=BatchSize,
            shuffle=False)

    TestGen = keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=preprocess_input_new).flow_from_directory(
            TestPath,
            target_size=(Height, Width),
            batch_size=BatchSize,
            shuffle=False)

    os.makedirs("models/h5/" + str(timestr), exist_ok=True)
    filepath = "models/h5/" + str(timestr) + "/" + "weights-improvement-{epoch:02d}-{val_accuracy:.4f}.hdf5"
    SaveModelImage(model, "models/h5/" + str(timestr) + "/" + "Graph.png")
    copyfile('face.py', "models/h5/" + str(timestr) + "/face.py")
    checkpoint = keras.callbacks.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    reduce_lr = keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.9, patience=5, min_lr=0.00001)
    ModelCallbacks = keras.callbacks.callbacks.LambdaCallback(
                            on_epoch_begin=None,
                            on_epoch_end=None,
                            on_batch_begin=None,
                            on_batch_end=None,
                            on_train_begin=None,
                            on_train_end=None)


    first = 100
    if not load_model:
        # data = model.fit_generator(
        #        generator = TrainGen,
        #        validation_data= ValidGen,
        #        epochs=first,
        #        callbacks=[ModelCallbacks, reduce_lr, checkpoint],
        #        verbose=1)
        models = [model]
        args = [{'generator':TrainGen,
                 'validation_data':TestGen,
                 'epochs':first,
                 'callbacks':[ModelCallbacks, reduce_lr],
                 'verbose':1}]

        ml = MLEXPS()
        ml.setTopic('Autism')
        ml.setCopyFileList(['face.py'])
        ml.setModels(models)
        ml.setArgList(args)
        ml.generator = True
        ml.saveBestOnly = False
        ml.startExprQ()
    else:
        model = load_model(model_path)

    Y_pred = model.predict_generator(TestGen)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(TestGen.classes, y_pred))
    print('Classification Report')
    target_names = ['Autistic', 'Non_Autistic']
    print(classification_report(TestGen.classes, y_pred, target_names=target_names))
