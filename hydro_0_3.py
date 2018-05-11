
# coding: utf-8

# In[1]:


import h5py
from keras.utils.generic_utils import Progbar
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import random
import numpy as np
import matplotlib
matplotlib.use('Pdf')
import matplotlib.pyplot as plt
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.optimizers import *

RND = 777
np.random.seed(RND)

GPU = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

OUT_DIR = 'A'
if not os.path.isdir(OUT_DIR): os.makedirs(OUT_DIR)

def read_particle_distribution(f): #f is a file
    array = []
    for i in range(4):
        next(f)
    for line in f: # read rest of lines
        array.append([float(x) for x in np.array(line.split())[[2,3]]])
    array = np.array(array)
    n_distribution = (plt.hist2d(array[:,0],array[:,1],bins=32,range=[[-2.5,2.5],[-2.5,2.5]])[0])[:,:,np.newaxis]
    return n_distribution #size:28*28

def read_initial_profile(f):
    array = []
    for i in range(8):
        next(f)
    for line in f: # read rest of lines
        array.append([float(x) for x in line.split()])
    array = np.array(array)[:,:,np.newaxis]
    return array

def read_all():
    path1 = './data/sample_particle'
    path2 = './data/initial_profile'
    files = os.listdir(path1)
    initials = []
    particles = []
    for file in files:
        f = open(path1 + '/' + file)
        g = open(path2 + '/' + file)
        n_distribution = read_particle_distribution(f)
        particles.append(n_distribution)
        intial_profile = read_initial_profile(g)
        initials.append(intial_profile)
        f.close()
        g.close()
    return initials,particles

def resize_with_resolution(resize,img):
    x = resize[1]
    y = resize[0]
    width = len(img[1])
    height = len(img[0])
    img2 = np.zeros((width*x,height*y))
    for i in range(width):
        for j in range(height):
            img2[x*i:x*(i+1),y*j:y*(j+1)] = img[i,j]
    return img2


initials,particles=read_all()

inputs = Input((261,261,1))


c1 = Conv2D(8, (2, 2), activation='relu', padding='valid') (inputs)
c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1) #260->130

c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2) #130->65

c3 = Conv2D(32, (2, 2), activation='relu', padding='valid') (p2)
c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)  #64->32

c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)  #32->16

c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)
p5 = MaxPooling2D(pool_size=(2, 2)) (c5)  #16->8

c51 = Conv2D(128, (3, 3), activation='relu', padding='same') (p5)
c51 = Conv2D(128, (3, 3), activation='relu', padding='same') (c51)
p51 = MaxPooling2D(pool_size=(2, 2)) (c51)  #8->4

c52 = Conv2D(128, (3, 3), activation='relu', padding='same') (p51)
c52 = Conv2D(128, (3, 3), activation='relu', padding='same') (c52) #4->4

u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c52)
u6 = concatenate([u6, c51])
c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)  #4->8

u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c5])
c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)  #8->16

u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c4])
c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)  #16->32


outputs = Conv2D(1, (1, 1), activation='relu') (c8)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer=RMSprop(lr=0.0001), loss='mse')
BATCH_SIZE = 10
epoch = 2

progress_bar = Progbar(target=epoch)

for i in range(epoch):
    index = np.random.choice(len(initials), BATCH_SIZE, replace=False)
    initial_batch = np.array(initials)[index]
    particle_batch = np.array(particles)[index]
    loss = model.train_on_batch(initial_batch, particle_batch)
    if i%100==0:
        img1 = model.predict(initials[89][np.newaxis,:,:,:])
        img2 = model.predict(initials[162][np.newaxis,:,:,:])
        plt.imsave(OUT_DIR + '/samples_1000_%07d.png' % i, resize_with_resolution((10,10),img1[0,:,:,0]), cmap=plt.cm.gray)
        plt.imsave(OUT_DIR + '/samples_1005_%07d.png' % i, resize_with_resolution((10,10),img2[0,:,:,0]), cmap=plt.cm.gray)
        progress_bar.update(i)

model.save('model.h5')

