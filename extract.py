import tempfile

# extract and plot each detected face in a photograph
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN
import math
from PIL import Image
import glob

margin = 0.1
saveDest = 'D:/Autism-Data/Facebook/RawFaces/Autistic/'
count = len(glob.glob(saveDest+'*'))

# draw each face separately
def draw_faces(filename, result_list):
    global count
    # load the image
    # plot each face as a subplot
    for i in range(len(result_list)):
        # get coordinates
        img = Image.open(filename)
        Iwidth, Iheight = img.size
        x1, y1, width, height = result_list[i]['box']
        x2, y2 = x1 + width, y1 + height

        x1 = max(math.floor(x1 * (1-margin)),0)
        y1 = max(math.floor(y1 * (1-margin)),0)
        x2 = min(math.floor(x2 * (1+margin)), Iwidth)
        y2 = min(math.floor(y2 * (1+margin)), Iheight)

        im_crop = img.crop((x1, y1, x2, y2)).save(saveDest + str(count) + '.jpg', quality=100)
        count += 1

# load image from file
# create the detector, using default weights
detector = MTCNN()

filestoMTCNN = glob.glob('D:/Autism-Data/Facebook/Raw/Autistic/*')
for file in filestoMTCNN:
    # detect faces in the image
    pixels = pyplot.imread(file)
    faces = detector.detect_faces(pixels)
    # display faces on the original image
    draw_faces(file, faces)
    pass
