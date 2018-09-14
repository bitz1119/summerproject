# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 14:57:30 2018
@author: bittoo
"""

import os
a = os.listdir(r"C:\Users\bittoo\Desktop\Coloring-greyscale-images-in-Keras\floydhub\Train")
import skimage
import matplotlib.pyplot as plt
from keras.preprocessing import image
from skimage.color import rgb2lab,lab2rgb
b = []
for i in range(len(a)):
    b.append(rgb2lab(skimage.img_as_float(image.load_img(a[i]))))

import numpy as np

b = np.asarray(b,'float')
X = []
for i in range(len(a)):
    X.append(b[i,:,:,:1])
X = np.asarray(X,'float')
plt.imshow(b[1,:,:,1])
    
    
Y = []
for i in range(len(a)):
    Y.append(b[i,:,:,1:])
Y = np.asarray(Y,'float')

X = X/128
Y = Y/128
    
x_test = X;
y_test = Y;
     
from keras.models import Sequential
from keras.layers import Conv2D,InputLayer,UpSampling2D
from keras.optimizers import rmsprop
model = Sequential()
model.add(InputLayer(input_shape=(None, None, 1)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))

# Finish model
model.compile(optimizer='rmsprop',loss='mean_squared_error')
model.fit(x=x_test, y=y_test, batch_size=1, epochs=500)    

y1 = model.predict(x_test)
ans = np.zeros([256,256,3])
y1 = y1*128;
x_test = x_test*128;
x_test.shape
ans[:,:,:1] = x_test[0]
ans[:,:,1:] = y1[0]

plt.imshow(ans)    
from skimage.color import lab2rgb
plt.imshow(lab2rgb(ans))    
a = lab2rgb(ans)
plt.imshow(ans[:,:,2])  
from skimage.io import imsave
lis = os.listdir()
for i in range(len(lis)):
    im = skimage.img_as_float(image.load_img(lis[i]))
    plt.imshow(im)
    test_im = np.zeros([1,256,256,1])
    test_im[0,:,:,:] = im[:,:,:1]
    test_out = model.predict(test_im)
    test_ans = np.zeros([256,256,3])
    test_ans[:,:,:1] =  im[:,:,:1]
    test_ans[:,:,1:] = test_out[0,:,:,:]
    plt.imshow(test_ans)
    test_ans= lab2rgb(test_ans)
    imsave(str(i)+'.jpg',test_ans)
    
"""  
from keras.models import load_model 
model.save('model_file.h5')
"""