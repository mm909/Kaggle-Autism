import keras
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from PIL import Image
import glob
import time
from shutil import copyfile
import os
from os import listdir
from os.path import isfile, join

BatchSize = 32
Height = 224
Width = 224
lr_rate=.0015


def SaveModelImage(Model, Title):
    keras.utils.vis_utils.plot_model(Model, to_file=Title, show_shapes=True, show_layer_names=True)
    return

def Summary(Model):
    print(Model.summary())
    return

def MakeModel(dlsize):
    BaseModel = VGGFace(model='senet50', include_top=False, input_shape=(Height, Width, 3), pooling='avg')
    last_layer = BaseModel.get_layer('avg_pool').output

    x = keras.layers.Flatten(name='flatten')(last_layer)
    # x = keras.layers.Dense(4096,activation='relu', name='fc1')(x)
    x = keras.layers.Dense(2048,activation='relu', name='fc1')(x)
    x = keras.layers.Dense(1024,activation='relu', name='fc2')(x)
    x = keras.layers.Dense(512,activation='relu', name='fc3')(x)
    # x = Dropout(rate=.4, seed=rand_seed)(x)
    out = keras.layers.Dense(2, activation='softmax', name='classifier')(x)
    DerivedModel = keras.Model(BaseModel.input, out)

    # LastLayer = BaseModel.layers[-1].output
    # tempModel = keras.layers.Dense(dlsize, activation='relu')(LastLayer)
    # tempModel = keras.layers.Dense(124, activation='relu')(tempModel)
    # tempModel = keras.layers.Dropout(rate = .4)(tempModel)
    # Predictions = keras.layers.Dense(2, activation='softmax')(tempModel)
    # DerivedModel = keras.Model(inputs = BaseModel.input, outputs = Predictions)

    for layer in DerivedModel.layers:
        layer.trainable = True
    # for layer in BaseModel.layers:
    #     layer.trainable = False

    DerivedModel.compile(keras.optimizers.Adam(lr=lr_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return DerivedModel

def preprocess_input_new(x):
    img = preprocess_input(keras.preprocessing.image.img_to_array(x))
    return keras.preprocessing.image.array_to_img(img)

def onEpochBegin(epoch, logs):
    print("Starting Epoch")
    return

if __name__ == "__main__":
    timestr = time.strftime("%Y%m%d-%H%M%S")
    model = MakeModel(1024)

    TrainPath = 'D:/Kaggle-Autism/data/train'
    ValidPath = 'D:/Kaggle-Autism/data/valid'
    TestPath  = 'D:/Kaggle-Autism/data/test'

    TrainGen = keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=preprocess_input_new,
            horizontal_flip=True,
            samplewise_center=True,
            rotation_range=20,
            zoom_range=0.05,
            shear_range=0.05,
            width_shift_range=.2,
            height_shift_range=.2,
            samplewise_std_normalization=True).flow_from_directory(
            TrainPath,
            target_size=(Height, Width),
            batch_size=BatchSize)

    ValidGen = keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=preprocess_input_new,
            samplewise_center=True,
            samplewise_std_normalization=True).flow_from_directory(
            ValidPath,
            target_size=(Height, Width),
            batch_size=BatchSize,
            shuffle=False)

    TestGen = keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=preprocess_input_new,
            samplewise_center=True,
            samplewise_std_normalization=True).flow_from_directory(
            TestPath,
            target_size=(Height, Width),
            batch_size=BatchSize,
            shuffle=False)

    os.makedirs("models/h5/" + str(timestr), exist_ok=True)
    filepath = "models/h5/" + str(timestr) + "/" + "weights-improvement-{epoch:02d}-{val_accuracy:.4f}.hdf5"
    checkpoint = keras.callbacks.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    reduce_lr = keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=5, min_lr=0.001)
    ModelCallbacks = keras.callbacks.callbacks.LambdaCallback(
                            on_epoch_begin=onEpochBegin,
                            on_epoch_end=None,
                            on_batch_begin=None,
                            on_batch_end=None,
                            on_train_begin=None,
                            on_train_end=None)

    first = 40
    data = model.fit_generator(
           generator = TrainGen,
           validation_data= ValidGen,
           epochs=first,
           callbacks=[ModelCallbacks, reduce_lr, checkpoint],
           verbose=1)

    model.save("models/h5/" + str(timestr) + "/" + str(round(data.history['val_accuracy'][-1],4)*100) + '.h5')

    path = 'D:/Kaggle-Autism/models/h5/'
    folders = [f for f in listdir(path) if join(path, f)]
    path += folders[-1] + '/'
    files = [f for f in listdir(path) if join(path, f)]
    files.pop()
    for file in files:
        if os.path.isfile(path +file): # this makes the code more robust
            os.remove(path+file)

    SaveModelImage(model, "models/h5/" + str(timestr) + "/" + "Graph.png")
    copyfile('face.py', "models/h5/" + str(timestr) + "/face.py")
