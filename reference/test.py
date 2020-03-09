import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Activation,Dropout,Conv2D, MaxPooling2D,BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model, Sequential
import numpy as np
import matplotlib.pyplot as plt
import time
import os

def get_paths(source_dir):
    test_path  = os.path.join(source_dir,'test')
    train_path = os.path.join(source_dir, 'train')
    valid_path = os.path.join(source_dir,'valid')
    classes = os.listdir(test_path)
    return [train_path, test_path, valid_path, classes]

def make_model(classes, lr_rate, height, width, model_size, rand_seed):
    size = len(classes)

    mobile = tf.keras.applications.mobilenet.MobileNet(include_top=True,
                                                       input_shape=(height,width,3),
                                                       pooling='avg', weights='imagenet',
                                                       alpha=1, depth_multiplier=1)

    x = mobile.layers[-6].output
    x = Dense(124, kernel_regularizer = regularizers.l2(l = 0.015), activation='relu')(x)
    x = Dropout(rate=.4, seed=rand_seed)(x)

    predictions = Dense(size, activation='softmax')(x)
    model = Model(inputs=mobile.input, outputs=predictions)
    for layer in model.layers:
        layer.trainable = True
    # for layer in model.layers[-80:]:
    #     layer.trainable=True


    model.compile(Adam(lr=lr_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model


def make_generators( paths, mode, batch_size, v_split, classes, height, width):
    file_names=[]
    labels=[]
    v_split=v_split/100.0

    train_gen=ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input,
            horizontal_flip=True,
            samplewise_center=True,
            width_shift_range=.2,
            height_shift_range=.2,
            validation_split=v_split,
            samplewise_std_normalization=True).flow_from_directory(
            paths[0],
            target_size=(height, width),
            batch_size=batch_size, seed=rand_seed)

    val_gen=ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input,
            samplewise_center=True,
            samplewise_std_normalization=True).flow_from_directory(
            paths[2],
            target_size=(height, width),
            batch_size=batch_size,
            seed=rand_seed,
            shuffle=False)

    test_gen=ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input,
            samplewise_center=True,
            samplewise_std_normalization=True).flow_from_directory(
            paths[1],
            target_size=(height, width),
            batch_size=batch_size,
            seed=rand_seed,
            shuffle=False)

    for file in test_gen.filenames:
        file_names.append(file)
    for label in test_gen.labels:
        labels.append(label)

    return [train_gen, test_gen, val_gen, file_names, labels]

def train(model, callbacks, train_gen, val_gen, epochs, start_epoch):
    start=time.time()
    data = model.fit_generator(
           generator = train_gen,
           validation_data= val_gen,
           epochs=epochs,
           initial_epoch=start_epoch,
           callbacks=callbacks,
           verbose=1)
    stop=time.time()

    duration = stop-start
    hrs=int(duration/3600)
    mins=int((duration-hrs*3600)/60)
    secs= duration-hrs*3600-mins*60
    msg='Training took\n {0} hours {1} minutes and {2:6.2f} seconds'
    print(msg.format(hrs, mins,secs))
    return data

def save_model(output_dir,subject, accuracy, height, width, model, weights):
    # save the model with the  subect-accuracy.h5
    acc=str(accuracy)[0:5]
    tempstr= subject + '-' + str(height) + '-' + str(width) + '-' + acc + '.h5'
    model.set_weights(weights)
    model_save_path = os.path.join(output_dir, tempstr)
    model.save(model_save_path)

def make_predictions(model, weights, test_gen, lr):
    config = model.get_config()
    pmodel = Model.from_config(config)  # copy of the model
    pmodel.set_weights(weights) #load saved weights with lowest validation loss
    pmodel.compile(Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    print('Training has completed. Now loading test set to see how accurate the model is')
    results = pmodel.evaluate(test_gen, verbose=0)
    print('Model accuracy on Test Set is {0:7.2f} %'.format(results[1]* 100))
    predictions = pmodel.predict(test_gen, verbose=0)
    return predictions

def tr_plot(tacc,vacc,tloss,vloss):
    #Plot the training and validation data
    Epoch_count=len(tloss)
    Epochs=[]
    for i in range (0,Epoch_count):
        Epochs.append(i+1)
    index_loss=np.argmin(vloss)#  this is the epoch with the lowest validation loss
    val_lowest=vloss[index_loss]
    index_acc=np.argmax(vacc)
    val_highest=vacc[index_acc]
    plt.style.use('fivethirtyeight')
    sc_label='best epoch= '+ str(index_loss+1)
    vc_label='best epoch= '+ str(index_acc + 1)
    fig,axes=plt.subplots(nrows=1, ncols=2, figsize=(15,5))
    axes[0].plot(Epochs,tloss, 'r', label='Training loss')
    axes[0].plot(Epochs,vloss,'g',label='Validation loss' )
    axes[0].scatter(index_loss+1,val_lowest, s=150, c= 'blue', label=sc_label)
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[1].plot (Epochs,tacc,'r',label= 'Training Accuracy')
    axes[1].plot (Epochs,vacc,'g',label= 'Validation Accuracy')
    axes[1].scatter(index_acc+1,val_highest, s=150, c= 'blue', label=vc_label)
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    plt.tight_layout
    #plt.style.use('fivethirtyeight')
    plt.show()

def display_pred(output_dir, pred, file_names, labels, subject, model_size,classes):
    trials=len(labels)
    errors=0
    for i in range (0,trials):
        p_class=pred[i].argmax()
        if p_class != labels[i]: #if the predicted class is not the same as the test label it is an error
            errors=errors + 1

    accuracy=100*(trials-errors)/trials
    return accuracy

def TF2_classify(source_dir, output_dir, mode, subject, v_split=5, epochs=20, batch_size=80,
                 lr_rate=.002, height=224, width=224, rand_seed=128, model_size='L'):
    model_size=model_size.upper()
    height=224
    width=224
    mode=mode.upper()
    paths=get_paths(source_dir)
    gens=make_generators(paths, mode, batch_size, v_split, paths[3], height, width)
    model=make_model(paths[3], lr_rate, height, width, model_size, rand_seed)

    # Dynamic learning rate for batches
    class tr(tf.keras.callbacks.Callback):
        best_weights=model.get_weights()
        best_acc=0
        patience=10
        p_count=0
        focus='acc'

        def __init__(self):
            super(tr, self).__init__()
            self.best_acc = 0
            self.patience=10
            self.p_count=0

        def on_batch_end(self, batch, logs=None):
            epoch=logs.get('epoch')
            acc=logs.get('accuracy')
            if tr.best_acc > .9:
                if tr.focus=='acc':
                    tr.focus='val'
            else:
                if tr.best_acc<acc:
                    tr.best_acc=acc
                    tr.p_count=0
                    tr.best_weights=model.get_weights()

                else:
                    tr.p_count=tr.p_count + 1
                    if tr.p_count >= tr.patience:
                        tr.p_count=0
                        lr=float(tf.keras.backend.get_value(self.model.optimizer.lr))
                        new_lr=lr*.99
                        tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)

    # Dynamic learning rate for epoches (Validaion)
    class val(tf.keras.callbacks.Callback):
        best_loss=np.inf
        best_weights=tr.best_weights
        lr=float(tf.keras.backend.get_value(model.optimizer.lr))

        def __init__(self):
            super(val, self).__init__()
            self.best_loss=np.inf
            self.best_weights=tr.best_weights
            self.lr=float(tf.keras.backend.get_value(model.optimizer.lr))

        def on_epoch_end(self, epoch, logs=None):
            v_loss=logs.get('val_loss')
            v_acc=logs.get('val_accuracy')

            if v_loss<val.best_loss:
                val.best_loss=v_loss
                val.best_weights=model.get_weights()
            else:
                if tr.focus=='val':
                        #validation loss did not improve at end of current epoch
                        lr=float(tf.keras.backend.get_value(self.model.optimizer.lr))
                        new_lr=lr * .7
                        tf.keras.backend.set_value(model.optimizer.lr, new_lr)
            val.lr=float(tf.keras.backend.get_value(model.optimizer.lr))

    callbacks=[tr(), val()]
    run_num=0
    tacc=[]
    tloss=[]
    vacc=[]
    vloss=[]
    start_epoch=0

    results = train(model, callbacks, gens[0], gens[2], epochs,start_epoch)

    # returns data from training the model - append the results for plotting
    tacc_new  = results.history['accuracy']
    tloss_new = results.history['loss']
    vacc_new  = results.history['val_accuracy']
    vloss_new = results.history['val_loss']
    for d in tacc_new:  # need to append new data from training to plot all epochs
        tacc.append(d)
    for d in tloss_new:
        tloss.append(d)
    for d in vacc_new:
        vacc.append(d)
    for d in vloss_new:
        vloss.append(d)

    tr_plot(tacc,vacc,tloss,vloss) # plot the data on loss and accuracy
    last_epoch=results.epoch[len(results.epoch)-1] # this is the last epoch run
    bestw=val.best_weights  # these are the saved weights with the lowest validation loss
    lr_rate=val.lr
    predictions=make_predictions(model, bestw, gens[1], lr_rate)
    accuracy=display_pred(output_dir, predictions, gens[3], gens[4], subject, model_size, paths[3])
    save_model(output_dir, subject, accuracy, height, width , model, bestw)


source_dir='data\\'
output_dir='data\\working\\'
subject='autism'
v_split=8
epochs=20
batch_size=16
lr_rate=.0015
height=224
width=224
rand_seed=100
model_size='L'
mode='sep'

TF2_classify(source_dir, output_dir, mode,subject, v_split= v_split, epochs=epochs,batch_size= batch_size,
         lr_rate= lr_rate,height=height, width=width,rand_seed=rand_seed, model_size=model_size)
