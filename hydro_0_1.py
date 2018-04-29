
# coding: utf-8

# In[91]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import keras.backend as K
from keras.layers import (Input, Dense, Reshape, Flatten, Lambda, merge,
                          Dropout, BatchNormalization, Activation, Embedding)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import (UpSampling2D, Conv2D, ZeroPadding2D,
                                        AveragePooling2D)
from keras.layers.local import LocallyConnected2D

from keras.models import Model, Sequential
import matplotlib
matplotlib.use['Pdf']
import matplotlib.pyplot as plt

import numpy as np
from keras.optimizers import *
from keras.utils.generic_utils import Progbar
from keras.models import load_model
from skimage import transform,data

RND = 777
np.random.seed(RND)

GPU = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

RUN = 'C'
OUT_DIR = 'out/' + RUN
TENSORBOARD_DIR = 'tensorboard/wgans/' + RUN
if not os.path.isdir(OUT_DIR): os.makedirs(OUT_DIR)
if not os.path.isdir(TENSORBOARD_DIR): os.makedirs(TENSORBOARD_DIR)

X_train = np.zeros((20,8,8,1))
for i in range(0,8):
    for j in range(0,8):
        if i==3 or i==4:
            if j>0 and j<7:
                X_train[:,i,j,:]+=1
        if j==3 or j==4:
            if i>0 and i<7:
                X_train[:,i,j,:]+=1
#plt.imshow(img[1,:,:], cmap='gray')
#plt.show()


K.set_image_dim_ordering('tf')


def generator(latent_size):
    
    latent = Input(shape=(latent_size,))
    x = Dense(2*2*8)(latent)
    x = Reshape((2,2,8))(x)
    x = UpSampling2D(size=(2,2))(x)
    
    x = ZeroPadding2D((1,1))(x)
    x = LocallyConnected2D(4, (3, 3), kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D(size=(2,2))(x)
    
    x = ZeroPadding2D((2,2))(x)
    x = LocallyConnected2D(4, (5, 5), kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    x = ZeroPadding2D((1,1))(x)
    x = LocallyConnected2D(1, (3, 3), kernel_initializer='glorot_normal')(x)
    x = Activation('relu')(x)
    
    return Model(inputs=latent, outputs=x)

def discriminator():
    
    image = Input(shape=(8,8,1))
    
    x = Conv2D(16, (4, 4), padding='same')(image)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Dropout(0.2)(x)
    
    x = ZeroPadding2D((1,1))(x)
    x = LocallyConnected2D(8, (3, 3), padding='valid', strides=(2,2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.2)(x)
    
    x = ZeroPadding2D((1,1))(x)
    x = LocallyConnected2D(8, (3, 3), padding='valid', strides=(2,2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)
    
    return Model(inputs=image, outputs=x)

def d_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

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

# save 10x10 sample of generated images


Z_SIZE = 20
    
D = discriminator()

D.compile(optimizer=RMSprop(lr=0.00005),loss=d_loss)

input_z = Input(shape=(Z_SIZE, ), name='input_z_')

G = generator(Z_SIZE)

# create combined D(G) model
output_is_fake = D(G(inputs=[input_z]))
DG = Model(inputs=input_z, outputs=output_is_fake)

DG.compile(optimizer=RMSprop(lr=0.00005),loss=d_loss)

def generate_samples(n=0, save=True):

    zz = np.random.normal(0., 1., (9, Z_SIZE))
    generated_images = G.predict(zz)

    rr = []
    for c in range(3):
        rr.append(
            np.concatenate(generated_images[c * 3:(1 + c) * 3]).reshape(24, 8))
    img = np.hstack(rr)
    img = resize_with_resolution((20,20),img)
    
    if save:
        plt.imsave(OUT_DIR + '/samples_%07d.png' % n, img, cmap=plt.cm.gray)

    return img

# write tensorboard summaries
sw = tf.summary.FileWriter(TENSORBOARD_DIR)
def update_tb_summary(step, write_sample_images=True):

    s = tf.Summary()

    # losses as is
    for names, vals in zip(('D_real_is_fake'
                            'D_fake_is_fake', 'DG_is_fake'),
                           (D_true_losses, D_fake_losses, DG_losses)):

        v = s.value.add()
        v.simple_value = vals[1]
        v.tag = names[0]

        v = s.value.add()
        v.simple_value = vals[2]
        v.tag = names[1]

    # D loss: -1*D_true_is_fake - D_fake_is_fake
    v = s.value.add()
    v.simple_value = -D_true_losses[-1] - D_fake_losses[-1]
    v.tag = 'D loss (-1*D_real_is_fake - D_fake_is_fake)'

    # generated image
    if write_sample_images:
        img = generate_samples(step, save=True)
        s.MergeFromString(tf.Session().run(
            tf.summary.image('samples_%07d' % step,
                             img.reshape([1, img.shape[0],img.shape[1], 1]))))

    sw.add_summary(s, step)
    sw.flush()
    
ITERATIONS = 4001
BATCH_SIZE = 10

progress_bar = Progbar(target=ITERATIONS)

DG_losses = []
D_true_losses = []
D_fake_losses = []


for it in range(ITERATIONS):

    if len(D_true_losses) > 0:
        progress_bar.update(
            it,
            values=[ # avg of 5 most recent
                    ('D_real_is_fake', np.mean(D_true_losses[-5:], axis=0)),
                    ('D_fake_is_fake', np.mean(D_fake_losses[-5:], axis=0)),
                    ('D(G)_is_fake', np.mean(DG_losses[-5:],axis=0)),
            ]
        )
        
    else:
        progress_bar.update(it)

    # 1: train D on real+generated images

    if (it % 1000) < 25 or it % 500 == 0: # 25 times in 1000, every 500th
        d_iters = 100
    else:
        d_iters = 5

    for d_it in range(d_iters):

        # unfreeze D
        D.trainable = True
        for l in D.layers: l.trainable = True

        # clip D weights

        for l in D.layers:
            weights = l.get_weights()
            weights = [np.clip(w, -0.01, 0.01) for w in weights]
            l.set_weights(weights)

        # 1.1: maximize D output on reals === minimize -1*(D(real))

        # draw random samples from real images
        index = np.random.choice(len(X_train), BATCH_SIZE, replace=False)
        real_images = X_train[index]

        D_loss = D.train_on_batch(real_images, -np.ones(BATCH_SIZE))
        D_true_losses.append(D_loss)

        # 1.2: minimize D output on fakes 

        zz = np.random.normal(0., 1., (BATCH_SIZE, Z_SIZE))
        generated_images = G.predict(zz)

        D_loss = D.train_on_batch(generated_images, np.ones(BATCH_SIZE))
        D_fake_losses.append(D_loss)

    # 2: train D(G) (D is frozen)
    # minimize D output while supplying it with fakes, 
    # telling it that they are reals (-1)

    # freeze D
    D.trainable = False
    for l in D.layers: l.trainable = False

    zz = np.random.normal(0., 1., (BATCH_SIZE, Z_SIZE)) 

    DG_loss = DG.train_on_batch(zz,-np.ones(BATCH_SIZE))

    DG_losses.append(DG_loss)

    if it % 10 == 0:
        update_tb_summary(it, write_sample_images=(it % 250 == 0))

#DG.save('DG.h5')

