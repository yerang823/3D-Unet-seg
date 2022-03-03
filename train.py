#-*- coding:utf-8 -*-

#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import glob, os
import tensorflow as tf
import keras 
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
from keras import backend as K
from keras.utils import plot_model
from keras.models import load_model
from keras.utils import multi_gpu_model

import cv2
from tqdm import tqdm, tqdm_notebook
from sklearn.model_selection import train_test_split

#Dila U-Net
from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import random

tf.logging.set_verbosity(tf.logging.INFO)
from tensorflow_large_model_support import LMS
    
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"


# ===========================================================
# Layers
# ===========================================================

def upsample_conv(filters, kernel_size, strides, padding):
    return Conv3DTranspose(filters, kernel_size, strides=strides, padding=padding)


def upsample_simple(filters, kernel_size, strides, padding):
    return UpSampling3D(strides)


# 많은 파라미터가 생기기 때문에 가장 중요한 피쳐맵을 뽑는 아랫단에만 넣는다는 개념
def Multi_dilated_conv(inputs, filters):

    dilated_conv1 = Conv3D(64, (1, 1, 1), strides=(1, 1, 1), activation = None, padding='same', kernel_initializer='he_normal')(inputs)
    dilated_conv1 = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), activation = None, dilation_rate = (2,2,2), padding='same', kernel_initializer='he_normal')(dilated_conv1) 
    dilated_conv1 = Conv3D(filters, (1, 1, 1), strides=(1, 1, 1), activation = None, padding='same', kernel_initializer='he_normal')(dilated_conv1)
    dilated_conv1_add = add([inputs, dilated_conv1])  ## 이부분이 숏컷 (residual)
    dilated_conv1_add = BatchNormalization(axis=4, scale=False)(dilated_conv1_add)  
    dilated_conv1_add = Activation(activation='relu')(dilated_conv1_add)
  
    dilated_conv2 = Conv3D(64, (1, 1, 1), strides=(1, 1, 1), activation = None, padding='same', kernel_initializer='he_normal')(inputs)  
    dilated_conv2 = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), activation = None, dilation_rate = (4,4,4), padding='same', kernel_initializer='he_normal')(dilated_conv2)
    dilated_conv2 = Conv3D(filters, (1, 1, 1), strides=(1, 1, 1), activation = None, padding='same', kernel_initializer='he_normal')(dilated_conv2)
    dilated_conv2_add = add([inputs, dilated_conv2])
    dilated_conv2_add = BatchNormalization(axis=4, scale=False)(dilated_conv2_add)
    dilated_conv2_add = Activation(activation='relu')(dilated_conv2_add)
    
    dilated_conv3 = Conv3D(64, (1, 1, 1), strides=(1, 1, 1), activation = None, padding='same', kernel_initializer='he_normal')(inputs)
    dilated_conv3 = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), activation = None, dilation_rate = (8,8,8), padding='same', kernel_initializer='he_normal')(dilated_conv3)
    dilated_conv3 = Conv3D(filters, (1, 1, 1), strides=(1, 1, 1), activation = None, padding='same', kernel_initializer='he_normal')(dilated_conv3)
    dilated_conv3_add = add([inputs, dilated_conv3])
    dilated_conv3_add = BatchNormalization(axis=4, scale=False)(dilated_conv3_add)
    dilated_conv3_add = Activation(activation='relu')(dilated_conv3_add)

    #dilated_conv4 = Conv3D(64, (1, 1, 1), strides=(1, 1, 1), activation = None, padding='same', kernel_initializer='he_normal')(inputs)
    #dilated_conv4 = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), activation = None, dilation_rate = (16,16,16), padding='same', kernel_initializer='he_normal')(dilated_conv4) ## 인풋 이미지가 128,128로 작기때문에 16까지는 필요없을 수 있다.
    #dilated_conv4 = Conv3D(filters, (1, 1, 1), strides=(1, 1, 1), activation = None, padding='same', kernel_initializer='he_normal')(dilated_conv4)
    #dilated_conv4_add = add([inputs, dilated_conv4])
    #dilated_conv4_add = BatchNormalization(axis=4, scale=False)(dilated_conv4_add)
    #dilated_conv4_add = Activation(activation='relu')(dilated_conv4_add)
    
  
    #pyramid = concatenate([dilated_conv1_add, dilated_conv2_add, dilated_conv3_add, dilated_conv4_add], axis=4)
    pyramid = concatenate([dilated_conv1_add, dilated_conv2_add, dilated_conv3_add], axis=4)
    return pyramid
  
#피쳐 뽑는 부분 or skip concat 부분 or RCL블락 통과 후 concat
def RCL_block(input, filedepth): 
    conv1 = Conv3D(filters=filedepth, kernel_size=[3, 3, 3], strides=(1, 1, 1), padding='same',activation='relu')(input)
    stack2 = BatchNormalization(axis=4, scale=False)(conv1)

    RCL = Conv3D(filters=filedepth, kernel_size=[3, 3, 3], strides=(1, 1, 1), padding='same', activation='relu')

    conv2 = RCL(stack2)
    stack3 = Add()([conv1, conv2])
    stack4 = BatchNormalization(axis=4, scale=False)(stack3)

    conv3 = Conv3D(filters=filedepth, kernel_size=[3, 3, 3], strides=(1, 1, 1), padding='same',activation='relu', weights=RCL.get_weights())(stack4) 
    ## RCL.get_weight 부분. 가중치를 레이어마다 공유하는것. 공부필요.
    stack5 =  Add()([conv1, conv3])
    stack6 = BatchNormalization(axis=4, scale=False)(stack5)

    conv4 = Conv3D(filters=filedepth, kernel_size=[3, 3, 3], strides=(1, 1, 1), padding='same',activation='relu', weights=RCL.get_weights())(stack6)
    stack7 =  Add()([conv1, conv4])
    stack8 = BatchNormalization(axis=4, scale=False)(stack7)


    return stack8

def Dense_U_Net(input_img, mode = 'tran', base = 32, scale = 2, num_classes = 7):
    if mode == 'tran':
        upsample = upsample_conv
    elif mode == 'simp':
        upsample = upsample_simple
    conv1 = Conv3D(base, 3, activation=None, padding='same', kernel_initializer='he_normal')(input_img)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation(activation='relu')(conv1)
    conv1 = Conv3D(base, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation(activation='relu')(conv1)
    
    conv1 = RCL_block(conv1, 32)
    pool1 = MaxPooling3D(pool_size=(1, 2, 2))(conv1)

    conv2 = Conv3D((scale) * base, 3, activation=None, padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation(activation='relu')(conv2)
    conv2 = Conv3D((scale) * base, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation(activation='relu')(conv2)
    
    conv2 = RCL_block(conv2, 64)
    pool2 = MaxPooling3D(pool_size=(1, 2, 2))(conv2)

    conv3 = Conv3D((scale * scale) * base, 3, activation=None, padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation(activation='relu')(conv3)
    conv3 = Conv3D((scale * scale) * base, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation(activation='relu')(conv3)
    
    conv3 = RCL_block(conv3, 128)
    pool3 = MaxPooling3D(pool_size=(1, 2, 2))(conv3)

    conv4 = Conv3D((scale * scale * scale) * base, 3, activation=None, padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation(activation='relu')(conv4)
    conv4 = Conv3D((scale * scale * scale) * base, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation(activation='relu')(conv4)
    
    conv4 = RCL_block(conv4, 256)
    pool4 = MaxPooling3D(pool_size=(1, 2, 2))(conv4)

    conv5 = Conv3D((scale * scale * scale * scale) * base, 3, activation=None, padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation(activation='relu')(conv5)
    conv5 = Conv3D((scale * scale * scale * scale) * base, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation(activation='relu')(conv5)
    
    #conv5 = RCL_block(conv5, 512)
    
    conv5 = Multi_dilated_conv(conv5, 512)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv3D((scale * scale * scale * scale) * base, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation(activation='relu')(conv5)
    
    pool6 = upsample((scale * scale * scale) * base, (2, 2, 2), strides=(1, 2, 2), padding='same')(conv5)
    merge = concatenate([conv4, pool6], axis = 4)
    conv6 = Conv3D((scale * scale * scale) * base, 3, activation=None, padding='same', kernel_initializer='he_normal')(merge)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation(activation='relu')(conv6)
    conv6 = Conv3D((scale * scale * scale) * base, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation(activation='relu')(conv6)

    pool7 = upsample((scale * scale) * base, (2, 2, 2), strides=(1, 2, 2), padding='same')(conv6)
    merge = concatenate([conv3, pool7], axis = 4)
    conv7 = Conv3D((scale * scale) * base, 3, activation=None, padding='same', kernel_initializer='he_normal')(merge)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation(activation='relu')(conv7)
    conv7 = Conv3D((scale * scale) * base, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation(activation='relu')(conv7)

    pool8 = upsample(scale * base, (2, 2, 2), strides=(1, 2, 2), padding='same')(conv7)
    merge = concatenate([conv2, pool8], axis = 4)
    conv8 = Conv3D(scale * base, 3, activation=None, padding='same', kernel_initializer='he_normal')(merge)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation(activation='relu')(conv8)
    conv8 = Conv3D(scale * base, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation(activation='relu')(conv8)

    pool9 = upsample(base, (2, 2, 2), strides=(1, 2, 2), padding='same')(conv8)
    merge = concatenate([conv1, pool9], axis = 4)
    conv9 = Conv3D(base, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge)
    out = Conv3D(num_classes, 3, activation='softmax', padding='same')(conv9)
    model = Model(inputs=input_img, outputs=out)
    
    return model

# ===========================================================
# loss 관련
# ===========================================================
def tversky_loss(y_true, y_pred):
    alpha = 0.5
    beta  = 0.5
    
    ones = K.ones(K.shape(y_true))
    p0 = y_pred      # proba that voxels are class i
    p1 = ones-y_pred # proba that voxels are not class i
    g0 = y_true
    g1 = ones-y_true
    
    num = K.sum(p0*g0, (0,1,2,3))
    den = num + alpha*K.sum(p0*g1,(0,1,2,3)) + beta*K.sum(p1*g0,(0,1,2,3))
    
    T = K.sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]
    
    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl-T

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_cost_0(y_true, y_predicted):

    mask_true = y_true[:, :, :, :, 0]
    mask_pred = y_predicted[:, :, :, :, 0]

    num_sum = 2.0 * K.sum(mask_true * mask_pred) + K.epsilon()
    den_sum = K.sum(mask_true) + K.sum(mask_pred)+ K.epsilon()

    return num_sum/den_sum



def dice_cost_1(y_true, y_predicted):

    mask_true = y_true[:, :, :, :, 1]
    mask_pred = y_predicted[:, :, :, :, 1]

    num_sum = 2.0 * K.sum(mask_true * mask_pred) + K.epsilon()
    den_sum = K.sum(mask_true) + K.sum(mask_pred)+ K.epsilon()

    return num_sum/den_sum


def dice_cost_1_loss(y_true, y_predicted):

    mask_true = y_true[:, :, :, :, 1]
    mask_pred = y_predicted[:, :, :, :, 1]

    num_sum = 2.0 * K.sum(mask_true * mask_pred) + K.epsilon()
    den_sum = K.sum(mask_true) + K.sum(mask_pred)+ K.epsilon()
    
    return 1-(num_sum/den_sum)


def dice_cost_01(y_true, y_predicted):

    dice_1 = dice_cost_0(y_true, y_predicted)
    dice_2 = dice_cost_1(y_true, y_predicted)

    return 1- (1/2*(dice_1+dice_2))


# Ref: salehi17, "Twersky loss function for image segmentation using 3D FCDN"
# -> the score is computed for each class separately and then summed
# alpha=beta=0.5 : dice coefficient
# alpha=beta=1   : tanimoto coefficient (also known as jaccard)
# alpha+beta=1   : produces set of F*-scores
# implemented by E. Moebel, 06/04/18
def tversky_loss(y_true, y_pred):
    alpha = 0.5
    beta  = 0.5
    
    ones = K.ones(K.shape(y_true))
    p0 = y_pred      # proba that voxels are class i
    p1 = ones-y_pred # proba that voxels are not class i
    g0 = y_true
    g1 = ones-y_true
    
    num = K.sum(p0*g0, (0,1,2,3))
    den = num + alpha*K.sum(p0*g1,(0,1,2,3)) + beta*K.sum(p1*g0,(0,1,2,3))
    
    T = K.sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]
    
    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl-T


'''
# ===========================================================
# Data load
# ===========================================================
nodule_num=15

print('%d Loading...'%nodule_num)
X = np.load('../data/manual/%d/241_CTs_resam.npy'%nodule_num, allow_pickle=True)
Y = np.load('../data/manual/%d/241_ytrain.npy'%nodule_num, allow_pickle=True)
print('Loaded.')
#X = np.load('/root/YR_Park/data/3d-unet/300_CTs_resam.npy', allow_pickle=True)
#Y = np.load('/root/YR_Park/data/3d-unet/300_ytrain.npy', allow_pickle=True)

train_dic, test_dic, train_label, test_label = train_test_split(X, Y, test_size=0.2)
test_dic, test_x, test_label, test_y = train_test_split(test_dic, test_label, test_size=0.5)


print(train_dic.shape, np.max(train_dic), np.min(train_dic))
print(train_label.shape, np.max(train_label), np.min(train_label))
print(test_dic.shape, np.max(test_dic), np.min(test_dic))
print(test_label.shape, np.max(test_label), np.min(test_label))
print(test_x.shape, np.max(test_x), np.min(test_x))
print(test_y.shape, np.max(test_y), np.min(test_y))

train_dic = train_dic.astype(np.float32)
test_dic = test_dic.astype(np.float32)
train_label = train_label.astype(np.float32)
test_label = test_label.astype(np.float32)
test_x = test_x.astype(np.float32)
test_y = test_y.astype(np.float32)


#np.save('/root/YR_Park/data/3d-unet/test_dic.npy',test_dic)
#np.save('/root/YR_Park/data/3d-unet/test_x.npy',test_x)
#np.save('/root/YR_Park/data/3d-unet/test_label.npy',test_label)
#np.save('/root/YR_Park/data/3d-unet/test_y.npy',test_y)

np.save('../data/manual/%d/test_dic.npy'%nodule_num,test_dic)
np.save('../data/manual/%d/test_x.npy'%nodule_num,test_x)
np.save('../data/manual/%d/test_label.npy'%nodule_num,test_label)
np.save('../data/manual/%d/test_y.npy'%nodule_num,test_y)


# ===========================================================
# Train
# ===========================================================

input_img = Input(shape=(128,128,128,1))
model = Dense_U_Net(input_img, mode = 'tran', base = 32, scale = 2, num_classes = 2)
model.summary()


lms_callback = LMS()#swapout_threshold=300, swapin_groupby=0, swapin_ahead=1)
lms_callback.batch_size = 2
lms_callback.autotune_plot= False


#model = multi_gpu_model(model, gpus=4)
model.compile(optimizer=Adam(lr=0.001), loss=tversky_loss,
              metrics=[dice_coef])
              #metrics=[dice_cost_0, dice_cost_1, dice_cost_01, tversky_loss])

checkpointer = ModelCheckpoint(filepath='../result/model/manual/3Dunet_best_test_%d.h5'%nodule_num, verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.000001,verbose=1)
earlystopper = EarlyStopping(patience=30, verbose=1)

#callbacks = [reduce_lr, earlystopper, checkpointer]

results = model.fit(train_dic,train_label,validation_data=(test_dic,test_label),verbose=1,
                    batch_size=1, epochs=500,
                    callbacks = [reduce_lr, earlystopper, checkpointer])
                    
fig = plt.figure()
# plt.axes(facecolor='white')
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('../result/model/manual/loss_test_%d.png'%nodule_num)
'''
