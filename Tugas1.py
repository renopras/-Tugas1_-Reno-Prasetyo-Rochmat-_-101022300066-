#Import necessary libraries 
import numpy as np
import tensorflow as tf
import keras
import cv2
from keras.layers import MaxPool2D,Conv2D,UpSampling2D,Input,Dropout
from keras.models import Sequential
from keras.preprocessing.image import img_to_array
import os
from tqdm import tqdm
import re
import matplotlib.pyplot as plt

#Getting landscape image data,resizing them and appending in array
from skimage import io


# to get the files in proper order
def sorted_alphanumeric(data):  
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)',key)]
    return sorted(data,key = alphanum_key)
# defining the size of the image
SIZE = 160
color_img = []
path = '../input/landscape-image-colorization/landscape Images/color'
files = os.listdir(path)
files = sorted_alphanumeric(files)
for i in tqdm(files):    
    if i == '6000.jpg':
        break
    else:    
        img = cv2.imread(path + '/'+i,1)
        # open cv reads images in BGR format so we have to convert it to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #resizing image
        img = cv2.resize(img, (SIZE, SIZE))
        img = img.astype('float32') / 255.0
        color_img.append(img_to_array(img))


gray_img = []
path = '../input/landscape-image-colorization/landscape Images/gray'
files = os.listdir(path)
files = sorted_alphanumeric(files)
for i in tqdm(files):
     if i == '6000.jpg':
        break
     else: 
        #img = cv2.imread(path + '/'+i,cv2.IMREAD_GRAYSCALE)
        img = io.imread(path + '/'+i)
        #resizing image
        img = cv2.resize(img, (SIZE, SIZE))
        img = img.astype('float32') / 255.0
        gray_img.append(img_to_array(img))
         
 #Plotting Color image and it's corresponding grayscale image
# defining function to plot images pair
def plot_images(color,grayscale):
    plt.figure(figsize=(15,15))
    plt.subplot(1,3,1)
    plt.title('Color Image', color = 'green', fontsize = 20)
    plt.imshow(color)
    plt.subplot(1,3,2)
    plt.title('Grayscale Image ', color = 'black', fontsize = 20)
    
    plt.imshow(grayscale.squeeze(), cmap='gray')
   
    plt.show()
#Plotting image pair
for i in range(3,10):
     plot_images(color_img[i],gray_img[i])
# Slicing and reshaping
train_gray_image = gray_img[:5500]
train_color_image = color_img[:5500]

test_gray_image = gray_img[5500:]
test_color_image = color_img[5500:]
# reshaping
train_g = np.reshape(train_gray_image,(len(train_gray_image),SIZE,SIZE,1))
train_c = np.reshape(train_color_image, (len(train_color_image),SIZE,SIZE,3))
print('Train color image shape:',train_c.shape)
print('Train gray image shape:',train_g.shape)

test_gray_image = np.reshape(test_gray_image,(len(test_gray_image),SIZE,SIZE,1))
test_color_image = np.reshape(test_color_image, (len(test_color_image),SIZE,SIZE,3))
print('Test color image shape',test_color_image.shape)
print('Test gray image shape',test_gray_image.shape)

#Defining our model
from keras import layers
def down(filters , kernel_size, apply_batch_normalization = True):
    downsample = tf.keras.models.Sequential()
    downsample.add(layers.Conv2D(filters,kernel_size,padding = 'same', strides = 2))
    if apply_batch_normalization:
        downsample.add(layers.BatchNormalization())
    downsample.add(keras.layers.LeakyReLU())
    return downsample


def up(filters, kernel_size, dropout = False):
    upsample = tf.keras.models.Sequential()
    upsample.add(layers.Conv2DTranspose(filters, kernel_size,padding = 'same', strides = 2))
    if dropout:
        upsample.dropout(0.2)
    upsample.add(keras.layers.LeakyReLU())
    return upsample
def model():
    inputs = layers.Input(shape= [160,160,1])
    d1 = down(128,(3,3),False)(inputs)
    d2 = down(128,(3,3),False)(d1)
    d3 = down(256,(3,3),True)(d2)
    d4 = down(512,(3,3),True)(d3)
    
    d5 = down(512,(3,3),True)(d4)
    #upsampling
    u1 = up(512,(3,3),False)(d5)
    u1 = layers.concatenate([u1,d4])
    u2 = up(256,(3,3),False)(u1)
    u2 = layers.concatenate([u2,d3])
    u3 = up(128,(3,3),False)(u2)
    u3 = layers.concatenate([u3,d2])
    u4 = up(128,(3,3),False)(u3)
    u4 = layers.concatenate([u4,d1])
    u5 = up(3,(3,3),False)(u4)
    u5 = layers.concatenate([u5,inputs])
    output = layers.Conv2D(3,(2,2),strides = 1, padding = 'same')(u5)
    return tf.keras.Model(inputs=inputs, outputs=output)
model = model()
model.summary()

#Fitting our model
from tensorflow.keras.callbacks import LearningRateScheduler
# Mokymo tempų funkcija
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.exp(-0.1).numpy()

# Callback'o sukūrimas
callback = LearningRateScheduler(scheduler)

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001), loss = 'mean_absolute_error',
              metrics = ['acc'], )

model.fit(train_g, train_c, epochs = 50,batch_size = 30,verbose = 1, callbacks=[callback])
model.save("gray_2_rgb_model.keras")

# load trained
model=keras.models.load_model('/kaggle/input/gray160x160x1_to_rgb160x160x3/keras/default/1/gray_2_rgb_model.keras')

# plotting colorized image along with grayscale and color image
# defining function to plot images pair
def plot_images(color,grayscale,predicted):
    plt.figure(figsize=(15,15))
    plt.subplot(1,3,1)
    plt.title('Color Image', color = 'green', fontsize = 20)
    plt.imshow(color)
    plt.subplot(1,3,2)
    plt.title('Grayscale Image ', color = 'black', fontsize = 20)
    plt.imshow(grayscale, cmap='gray')
    plt.subplot(1,3,3)
    plt.title('Predicted Image ', color = 'Red', fontsize = 20)
    plt.imshow(predicted)
   
    plt.show()

for i in range(50,58):
    predicted = np.clip(model.predict(test_gray_image[i].reshape(1,SIZE, SIZE,1)),0.0,1.0).reshape(SIZE, SIZE,3)
    plot_images(test_color_image[i],test_gray_image[i].squeeze(),predicted)

 from skimage import io
import cv2
#image_rgb = io.imread('https://www.vle.lt/uploads/_CGSmartImage/114292_1-1608eda0d9c54043ad61ed96f6239ef6.jpg')
image_rgb = io.imread('https://upload.wikimedia.org/wikipedia/commons/thumb/2/24/Babylone_3ans1.jpg/375px-Babylone_3ans1.jpg')

img_rgb = cv2.resize(image_rgb, (160, 160))
img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    # Normalize pixel values
img_array = img.astype(np.float32) / 255.0  



    # Make predictions
predicted = np.clip(model.predict(img_array.reshape(1,SIZE, SIZE,1)),0.0,1.0).reshape(SIZE, SIZE,3)
plot_images(img_rgb,img_array,predicted)
