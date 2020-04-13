import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

alpha = 0.8


strings = []
strings.append('Predicted\ncondition')
strings.append('Total\npopluation')
strings.append('Predicted\ncondition\npositive')
strings.append('Predicted\ncondition\nnegative')

strings.append('True condition')
strings.append('Condition positive')
strings.append('True positive')
strings.append('False negative')
strings.append('True positive rate (TPR)\nSensitivity')
strings.append('False negative rate (FNR)')

strings.append('Condition negative')
strings.append('False positive')
strings.append('True negative')
strings.append('False positive rate (FPR)')
strings.append('True negative rate (TNR)\nSpecificity')

strings.append('Prevalence')
strings.append('Precision')
strings.append('False omission rate')
strings.append('Positive likelihood ratio')
strings.append('Negative likelihood ratio')

strings.append('Accuracy')
strings.append('False discovery rate')
strings.append('Negative predicitve value')

strings.append('Diagnostic\nodds ratio')
strings.append('F1 score')

fig, ax = plt.subplots()

rects = []
rects.append(Rectangle((0,0.7), 0.10, 0.20, facecolor='#bbeeee', alpha=alpha, transform=ax.transAxes))
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

rects.append(Rectangle((0.5,.9), 0.15, 0.05, facecolor='#eeeecc', alpha=alpha, transform=ax.transAxes))
rects.append(Rectangle((0.5,0.8), 0.15, 0.1, facecolor='#ccffee', alpha=alpha, transform=ax.transAxes))
rects.append(Rectangle((0.5,0.7), 0.15, 0.1, facecolor='#eeddee', alpha=alpha, transform=ax.transAxes))
rects.append(Rectangle((0.5,0.6), 0.15, 0.1, facecolor='#eeeeee', alpha=alpha, transform=ax.transAxes))
rects.append(Rectangle((0.5,0.5), 0.15, 0.1, facecolor='#cccccc', alpha=alpha, transform=ax.transAxes))

rects.append(Rectangle((0.65,.9), 0.15, 0.05, facecolor='#cbeecc', alpha=alpha, transform=ax.transAxes))
rects.append(Rectangle((0.65,0.8), 0.15, 0.1, facecolor='#cceeff', alpha=alpha, transform=ax.transAxes))
rects.append(Rectangle((0.65,0.7), 0.15, 0.1, facecolor='#aaddcc', alpha=alpha, transform=ax.transAxes))

rects.append(Rectangle((0.65,0.5), 0.075, 0.2, facecolor='#dddddd', alpha=alpha, transform=ax.transAxes))
rects.append(Rectangle((0.725,0.5), 0.075, 0.2, facecolor='#dcffdd', alpha=alpha, transform=ax.transAxes))

for i, string in enumerate(strings):
    ax.add_patch(rects[i])
    ax.add_artist(rects[i])
    rx, ry = rects[i].get_xy()
    cx = rx + rects[i].get_width() / 2.0
    cy = ry + rects[i].get_height() / 2.0
    ax.annotate(string, (cx, cy), color='black', fontsize=10, ha='center', va='center')

plt.axis('off')
plt.show()
