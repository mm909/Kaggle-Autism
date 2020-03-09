import keras
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from PIL import Image
import glob

BatchSize = 32
Height = 224
Width = 224

def SaveModelImage(Model, Title):
    keras.utils.vis_utils.plot_model(Model, to_file=Title, show_shapes=True, show_layer_names=True)
    return

def Summary(Model):
    print(Model.summary())
    return

def MakeModel(size, dlsize):
    BaseModel = VGGFace(model='resnet50', include_top=False, input_shape=(Height, Width, 3), pooling='avg')

    LastLayer = BaseModel.layers[-1].output
    tempModel = keras.layers.Dense(dlsize, activation='relu')(LastLayer)
    tempModel = keras.layers.Dense(124, activation='relu')(tempModel)
    tempModel = keras.layers.Dropout(rate = .2)(tempModel)
    Predictions = keras.layers.Dense(size, activation='softmax')(tempModel)
    DerivedModel = keras.Model(inputs = BaseModel.input, outputs = Predictions)

    for layer in DerivedModel.layers:
        layer.trainable = True

    DerivedModel.compile(keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return DerivedModel

if __name__ == "__main__":
    model = MakeModel(2, 1024)

    TrainPath = 'D:/Kaggle-Autism/data/train'
    ValidPath = 'D:/Kaggle-Autism/data/valid'
    TestPath  = 'D:/Kaggle-Autism/data/test'

    TrainGen = keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=preprocess_input,
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
            preprocessing_function=preprocess_input,
            samplewise_center=True,
            samplewise_std_normalization=True).flow_from_directory(
            ValidPath,
            target_size=(Height, Width),
            batch_size=BatchSize,
            shuffle=False)

    TestGen = keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=preprocess_input,
            samplewise_center=True,
            samplewise_std_normalization=True).flow_from_directory(
            TestPath,
            target_size=(Height, Width),
            batch_size=BatchSize,
            shuffle=False)

    data = model.fit_generator(
           generator = TrainGen,
           validation_data= ValidGen,
           epochs=5,
           verbose=1)
