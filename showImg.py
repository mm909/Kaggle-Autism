from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

rowCount = 5
colCount = 5
index = 0

fig = plt.figure()

gs1 = gridspec.GridSpec(rowCount, colCount)
gs1.update(wspace=0.0, hspace=0.05) # set the spacing between axes.


print(f"{1:04d}")

for row in range(rowCount):
    for col in range(colCount):
        img = mpimg.imread(f'D:/Autism-Data/Kaggle/v5/consolidated/Non_Autistic/{index+1:04d}.jpg')
        ax1 = plt.subplot(gs1[index])
        # ax1 = fig.add_subplot(rowCount,colCount,index)
        ax1.axis('off')
        ax1.set_aspect('equal')
        ax1.imshow(img)
        index += 1

# img = mpimg.imread('D:/Autism-Data/Kaggle/v5/consolidated/Non_Autistic/0681.jpg')
# ax2 = fig.add_subplot(3,3,2)
# ax2.imshow(img)
# ax2.axis('off')
#
# img = mpimg.imread('D:/Autism-Data/Kaggle/v5/consolidated/Autistic/0311.jpg')
# ax3 = fig.add_subplot(3,3,3)
# ax3.imshow(img)
# ax3.axis('off')

# img = mpimg.imread('D:/Autism-Data/Kaggle/v5/consolidated/Non_Autistic/0003.jpg')
# ax3 = fig.add_subplot(2,2,3)
# ax3.imshow(img)
# ax3.axis('off')
#
# img = mpimg.imread('D:/Autism-Data/Kaggle/v5/consolidated/Non_Autistic/0002.jpg')
# ax4 = fig.add_subplot(2,2,4)
# ax4.imshow(img)
# ax4.axis('off')

plt.tight_layout()
plt.show()
