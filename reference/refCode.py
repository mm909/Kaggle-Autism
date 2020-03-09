
# https://www.kaggle.com/vaibhavsxn/an-apple-a-day
image_gen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees
                               width_shift_range=0.10, # Shift the pic width by a max of 5%
                               height_shift_range=0.10, # Shift the pic height by a max of 5%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)
                               zoom_range=0.1, # Zoom in by 10% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                              )

image_gen.flow_from_directory(my_data_dir)

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(patience=2)

model = load_model('/kaggle/input/trained/fruit.h5')
from tensorflow.keras.preprocessing import image
carambola1 = carambola+'/'+os.listdir(carambola)[15]
my_image = image.load_img(carambola1,target_size=image_shape)
my_image = image.img_to_array(my_image)
my_image = np.expand_dims(my_image, axis=0)
model.predict(my_image)
