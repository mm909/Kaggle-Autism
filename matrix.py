import keras
import numpy as np
from keras.models import load_model
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

# model_path = 'D:/Kaggle-Autism/models/h5/20200311-135911/weights-improvement-44-0.8500.hdf5'
# test_path  = 'D:/Autism-Data/Kaggle/v5/test'
#
# height = 224
# width  = 224
# batch_size = 32
#
# def preprocess_input_new(x):
#     img = preprocess_input(keras.preprocessing.image.img_to_array(x), version = 2)
#     return keras.preprocessing.image.array_to_img(img)
#
# test_gen = keras.preprocessing.image.ImageDataGenerator(
#         preprocessing_function=preprocess_input_new).flow_from_directory(
#         test_path,
#         target_size=(height, width),
#         batch_size=batch_size,
#         shuffle=False)
#
# model = load_model(model_path)
# Y_pred = model.predict_generator(test_gen)
# y_pred = np.argmax(Y_pred, axis=1)
# print('Confusion Matrix')
# print(confusion_matrix(test_gen.classes, y_pred))
# print('Classification Report')
# target_names = ['Autistic', 'Non_Autistic']
# print(classification_report(test_gen.classes, y_pred, target_names=target_names))

matrix = np.array([[112,38],[16,134]])
print(matrix)

TPop = matrix.sum()
TP = matrix[0][0]
FP = matrix[0][1]
FN = matrix[1][0]
TN = matrix[1][1]
CP = TP + FN
CN = FP + TN
PCP = TP + FP
PCN = FN + TN
TPR = round(TP/CP,2)
FNR = round(FN/CP,2)
FPR = round(FP/CN,2)
TNR = round(TN/CN,2)
prevalence = round(CP/TP,2)
PPV = round(TP/PCP,2)
FOR = round(FN/PCN,2)
PLR = round(TPR/FPR,2)
NLR = round(FNR/TNR,2)
ACC = round((TP + TN) / TPop,2)
FDR = round(FP/PCP,2)
NPV = round(TN/PCN,2)
DOR = round(PLR/NLR,2)
F1 = round(2 * ((PPV * TPR)/(PPV + TPR)),2)

alpha = 0.8

strings = []
strings.append('Predicted\ncondition')
strings.append('Total popluation\n'+ str(TPop))
strings.append('Predicted\ncondition\npositive')
strings.append('Predicted\ncondition\nnegative')

strings.append('True condition')
strings.append('Condition positive')
strings.append('True positive\n'+ str(TP))
strings.append('False negative\n'+ str(FN))
strings.append('True positive rate (TPR)\nSensitivity\n'+ str(TPR))
strings.append('False negative rate (FNR)\n'+ str(FNR))

strings.append('Condition negative')
strings.append('False positive\n'+ str(FP))
strings.append('True negative\n'+ str(TN))
strings.append('False positive rate (FPR)\n'+ str(FPR))
strings.append('True negative rate (TNR)\nSpecificity\n'+ str(TNR))

# strings.append('Prevalence\n'+ str(prevalence))
# strings.append('Precision\n'+ str(PPV))
# strings.append('False omission rate\n'+ str(FOR))
# strings.append('Positive likelihood ratio\n'+ str(PLR))
# strings.append('Negative likelihood ratio\n'+ str(NLR))
#
# strings.append('Accuracy\n'+ str(ACC))
# strings.append('False discovery rate\n'+ str(FDR))
# strings.append('Negative predicitve value\n'+ str(NPV))
#
# strings.append('Diagnostic\nodds ratio\n'+ str(DOR))
# strings.append('F1 score\n'+ str(F1))

fig, ax = plt.subplots()

rects = []
rects.append(Rectangle((0.05,0.7), 0.05, 0.20, facecolor='#bbeeee', alpha=alpha, transform=ax.transAxes))
rects.append(Rectangle((0.1,0.9), 0.10, 0.05, facecolor='#dddddd', alpha=alpha, transform=ax.transAxes))
rects.append(Rectangle((0.1,0.8), 0.10, 0.10, facecolor='#ccffff', alpha=alpha, transform=ax.transAxes))
rects.append(Rectangle((0.1,0.7), 0.10, 0.10, facecolor='#aadddd', alpha=alpha, transform=ax.transAxes))

rects.append(Rectangle((0.2,.95), 0.3, 0.05, facecolor='#edeebb', alpha=alpha, transform=ax.transAxes))
rects.append(Rectangle((0.2,.9), 0.15, 0.05, facecolor='#ffffcc', alpha=alpha, transform=ax.transAxes))
rects.append(Rectangle((0.2,0.8), 0.15, 0.1, facecolor='#cbffcc', alpha=alpha, transform=ax.transAxes))
rects.append(Rectangle((0.2,0.7), 0.15, 0.1, facecolor='#ffdddd', alpha=alpha, transform=ax.transAxes))
rects.append(Rectangle((0.2,0.6), 0.15, 0.1, facecolor='#eeffcc', alpha=alpha, transform=ax.transAxes))
rects.append(Rectangle((0.2,0.5), 0.15, 0.1, facecolor='#ffeecc', alpha=alpha, transform=ax.transAxes))

rects.append(Rectangle((0.35,.9), 0.15, 0.05, facecolor='#dcddaa', alpha=alpha, transform=ax.transAxes))
rects.append(Rectangle((0.35,0.8), 0.15, 0.1, facecolor='#eedddd', alpha=alpha, transform=ax.transAxes))
rects.append(Rectangle((0.35,0.7), 0.15, 0.1, facecolor='#baeebb', alpha=alpha, transform=ax.transAxes))
rects.append(Rectangle((0.35,0.6), 0.15, 0.1, facecolor='#eeddbb', alpha=alpha, transform=ax.transAxes))
rects.append(Rectangle((0.35,0.5), 0.15, 0.1, facecolor='#ddeebb', alpha=alpha, transform=ax.transAxes))

# rects.append(Rectangle((0.5,.9), 0.15, 0.05, facecolor='#eeeecc', alpha=alpha, transform=ax.transAxes))
# rects.append(Rectangle((0.5,0.8), 0.15, 0.1, facecolor='#ccffee', alpha=alpha, transform=ax.transAxes))
# rects.append(Rectangle((0.5,0.7), 0.15, 0.1, facecolor='#eeddee', alpha=alpha, transform=ax.transAxes))
# rects.append(Rectangle((0.5,0.6), 0.15, 0.1, facecolor='#eeeeee', alpha=alpha, transform=ax.transAxes))
# rects.append(Rectangle((0.5,0.5), 0.15, 0.1, facecolor='#cccccc', alpha=alpha, transform=ax.transAxes))
#
# rects.append(Rectangle((0.65,.9), 0.15, 0.05, facecolor='#cbeecc', alpha=alpha, transform=ax.transAxes))
# rects.append(Rectangle((0.65,0.8), 0.15, 0.1, facecolor='#cceeff', alpha=alpha, transform=ax.transAxes))
# rects.append(Rectangle((0.65,0.7), 0.15, 0.1, facecolor='#aaddcc', alpha=alpha, transform=ax.transAxes))
#
# rects.append(Rectangle((0.65,0.5), 0.075, 0.2, facecolor='#dddddd', alpha=alpha, transform=ax.transAxes))
# rects.append(Rectangle((0.725,0.5), 0.075, 0.2, facecolor='#dcffdd', alpha=alpha, transform=ax.transAxes))

for i, string in enumerate(strings):
    ax.add_patch(rects[i])
    ax.add_artist(rects[i])
    rx, ry = rects[i].get_xy()
    cx = rx + rects[i].get_width() / 2.0
    cy = ry + rects[i].get_height() / 2.0
    ax.annotate(string, (cx, cy), color='black', fontsize=10, ha='center', va='center')

plt.axis('off')
plt.show()
