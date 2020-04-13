import os
from os import listdir
from os.path import isfile, join
import time
from shutil import copyfile
import keras
from keras.utils import plot_model
import matplotlib.pyplot as plt

class MLEXPS:

    def __init__(self):
        print('MLEXPS v3')
        self.topic = 'TOPIC'
        self.baseFolder = 'experiments'
        self.exprTimeStamp = 0
        self.exprFilePath = ''
        self.exprWeightPath = ''
        self.copyFileList = []
        self.currModel = None
        self.currArgs = None
        self.models = []
        self.argList = []
        self.generator = False
        return

    def startExprQ(self):
        if(len(self.models) != len(self.argList)):
            print("Models and Args do not match up.")
            return
        print("Length of queue:", len(self.models))
        for i, expr in enumerate(self.models):
            self.setCurrModel(expr)
            self.setCurrArgs(self.argList[i])
            self.startExpr()
            pass
        return

    def startExpr(self):
        self.currModel.summary()
        self.setupExprDir()
        checkpoint = keras.callbacks.callbacks.ModelCheckpoint(self.exprWeightPath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        csv = keras.callbacks.callbacks.CSVLogger(self.exprFilePath + '/logs/training/csvlog.csv', separator=',')
        cb = [checkpoint, csv]
        if 'callbacks' in self.currArgs:
            self.currArgs['callbacks'].append(checkpoint)
            self.currArgs['callbacks'].append(csv)
        else:
            self.currArgs['callbacks'] = [checkpoint, csv]
        if self.generator:
            history = self.currModel.fit_generator(**self.currArgs)
        else:
            history = self.currModel.fit(**self.currArgs)
        self.saveFigures(history)

        self.cleanUpWeights()
        return

    def saveFigures(self, history):

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(self.exprFilePath + '/logs/training/accuracy.png')
        plt.close()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='lower right')
        plt.savefig(self.exprFilePath + '/logs/training/loss.png')
        plt.close()

        plt.rcParams['figure.figsize'] = [10,5]
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Model Stats')
        ax1.set_title("Model Accuracy")
        ax1.set(xlabel='Epoch', ylabel='Accuracy')
        ax1.plot(history.history['accuracy'])
        ax1.plot(history.history['val_accuracy'])
        ax1.legend(['Train', 'Test'], loc='lower right')
        ax2.set_title("Model Loss")
        ax2.set(xlabel='Epoch', ylabel='Loss')
        ax2.plot(history.history['loss'])
        ax2.plot(history.history['val_loss'])
        ax2.legend(['Train', 'Test'], loc='upper right')
        plt.savefig(self.exprFilePath + '/logs/training/combined.png')
        plt.close()

        for key, value in history.history.items():
            with open(self.exprFilePath + '/logs/training/' + key + ".csv", 'w') as file:
                [file.write(str(num)+",") if i != len(value) - 1 else file.write(str(num)) for i, num in enumerate(value)]


        return

    def copyFiles(self):
        for file in self.copyFileList:
            copyfile(file, self.baseFolder + "/" + self.topic + "/" + str(self.exprTimeStamp) + '/files' + "/" + file)
        return

    def setupExprDir(self):
        self.exprTimeStamp = time.strftime("%Y%m%d-%H%M%S")

        os.makedirs(self.baseFolder + "/" + self.topic + "/" + str(self.exprTimeStamp), exist_ok=True)
        self.exprFilePath = self.baseFolder + "/" + self.topic + "/" + str(self.exprTimeStamp)
        os.makedirs(self.exprFilePath + '/weights', exist_ok=True)
        os.makedirs(self.exprFilePath + '/logs', exist_ok=True)
        os.makedirs(self.exprFilePath + '/logs/model', exist_ok=True)
        os.makedirs(self.exprFilePath + '/logs/training', exist_ok=True)
        os.makedirs(self.exprFilePath + '/files', exist_ok=True)

        self.exprWeightPath = self.exprFilePath + '/weights' + "/" + "weights-improvement-{epoch:02d}-{val_accuracy:.4f}.hdf5"
        self.copyFiles()

        if(self.currModel):
            with open(self.baseFolder + "/" + self.topic + "/" + str(self.exprTimeStamp) + '/logs/model' + '/summary.txt', 'w') as file:
                self.currModel.summary(print_fn=lambda x: file.write(x + '\n'))
        plot_model(self.currModel, to_file=self.exprFilePath + '/logs/model/model.png')
        return

    def cleanUpWeights(self):
        files = [f for f in listdir(self.exprFilePath + '/weights') if join(self.exprFilePath + '/weights', f)]
        files.pop()
        for file in files:
            if os.path.isfile(self.exprFilePath + '/weights/' + file) and os.path.splitext(file)[1] == '.hdf5':
                os.remove(self.exprFilePath + '/weights/' + file)
        return

    def setModels(self, models):
        self.models = models
        return

    def setCurrModel(self, model):
        self.currModel = model
        return

    def setArgList(self, argList):
        self.argList = argList
        return

    def setCurrArgs(self, currArgs):
        self.currArgs = currArgs
        return

    def setTopic(self, topic):
        self.topic = topic
        return

    def addCopyFile(self, file):
        self.copyFileList.append(file)
        return

    def setCopyFileList(self, files):
        self.copyFileList = files
        return
