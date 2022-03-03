#-*- coding:utf-8 -*-

#!/usr/bin/env python

import os 
import numpy as np
import pandas as pd
import glob

from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.models import *
from keras.optimizers import *
from keras import backend as K
from keras.layers.core import Lambda
from keras.layers.merge import concatenate,add
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers import Input, merge, Dropout, concatenate, Conv3D, MaxPooling3D, UpSampling3D, Activation, Conv3DTranspose, Reshape, multiply
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, History, ReduceLROnPlateau, EarlyStopping
from keras.utils.generic_utils import get_custom_objects
from keras.utils import multi_gpu_model
from keras.models import Model
from keras.regularizers import l2

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"


# ===========================================================
# Layers
# ===========================================================

def Conv3D_block(input_layer, out_n_filters, kernel_size=[3,3,3], stride=[1,1,1], padding='same'):
    
    layer = input_layer
    for i in range(2):
        layer = Conv3D(out_n_filters, kernel_size, strides=stride, padding=padding)(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)        
        
    out_layer = layer
    
    return out_layer

def Up_and_Concate(down_layer, layer):
    
    input_channel = down_layer.get_shape().as_list()[4]
    output_channel = input_channel // 2
    up = UpSampling3D(size = (2,2,2))(down_layer)
    concate = concatenate([up, layer])
    return concate
    
    
def attention_block_3d(x, g, inter_channel):

    theta_x = Conv3D(inter_channel, [1, 1, 1], strides=[1, 1, 1])(x)
    phi_g = Conv3D(inter_channel, [1, 1, 1], strides=[1, 1, 1])(g)
    f = Activation('relu')(add([theta_x, phi_g]))
    psi_f = Conv3D(1, [1, 1, 1], strides=[1, 1, 1])(f)
    rate = Activation('sigmoid')(psi_f)
    att_x = multiply([x, rate])

    return att_x
    
    
def attention_up_and_concate(down_layer, layer):
    in_channel = down_layer.get_shape().as_list()[4]
    up = UpSampling3D((2, 2, 2))(down_layer)
    layer = attention_block_3d(x=layer, g=up, inter_channel=in_channel // 4)
    concate = concatenate([up, layer])
    return concate

def U_Net_3D(time, img_w, img_h, n_label):
        
    inputs = Input((time, img_w, img_h, 1), dtype = 'float32')
    x = inputs
    depth = 4
    features = 32
    down_layer = []
    encoder_featuremap = []
    
    for i in range(depth):
        
        x = Conv3D_block(x, features)
        down_layer.append(x)
        x = MaxPooling3D(pool_size=[2, 2, 2], strides=[2, 2, 2])(x)
        features = features * 2
        
    x = Conv3D_block(x, features)
    
    for i in reversed(range(depth)):
        features = features // 2
        x = attention_up_and_concate(x, down_layer[i])
        x = Conv3D_block(x, features)
        encoder_featuremap.append(x)    
       
#     output_1 = UpSampling3D((8, 8, 8))(encoder_featuremap[0])
#     output_1 = Conv3D(n_label, 3, activation='sigmoid', padding='same', name='out_1')(output_1)

    output_2 = UpSampling3D((4, 4, 4))(encoder_featuremap[1])
    output_2 = Conv3D(n_label, 3, activation='sigmoid', padding='same', name='out_2')(output_2)
    
    output_3 = UpSampling3D((2, 2, 2))(encoder_featuremap[2])
    output_3 = Conv3D(n_label, 3, activation='sigmoid',padding='same', name='out_3')(output_3) 
    
#     output_4 = UpSampling3D(features, (1, 1, 1))(encoder_featuremap[3])
    output_4 = Conv3D(n_label, 3, activation='sigmoid', padding='same', name='out_4')(encoder_featuremap[3]) 
    
    model = Model(inputs = inputs, outputs = [output_2, output_3, output_4])
    model.summary()
    
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

epsilon = K.epsilon()
gamma = 0
alpha = 0.6
beta = 0.6

  
def recall(y_target, y_pred):
    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
    y_target_yn = K.round(K.clip(y_target, 0, 1)) # 실제값을 0(Negative) 또는 1(Positive)로 설정한다
    y_pred_yn = K.round(K.clip(y_pred, 0, 1)) # 예측값을 0(Negative) 또는 1(Positive)로 설정한다

    count_true_positive = K.sum(y_target_yn * y_pred_yn) 

    count_true_positive_false_negative = K.sum(y_target_yn)

    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    recall = count_true_positive / (count_true_positive_false_negative + K.epsilon())

    # return a single tensor value
    return recall


def precision(y_target, y_pred):
    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
    y_pred_yn = K.round(K.clip(y_pred, 0, 1)) # 예측값을 0(Negative) 또는 1(Positive)로 설정한다
    y_target_yn = K.round(K.clip(y_target, 0, 1)) # 실제값을 0(Negative) 또는 1(Positive)로 설정한다
    count_true_positive = K.sum(y_target_yn * y_pred_yn) 

    count_true_positive_false_positive = K.sum(y_pred_yn)
    
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    precision = count_true_positive / (count_true_positive_false_positive + K.epsilon())

    # return a single tensor value
    return precision


def f1score(y_target, y_pred):
    _recall = recall(y_target, y_pred)
    _precision = precision(y_target, y_pred)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    _f1score = ( 2 * _recall * _precision) / (_recall + _precision+ K.epsilon())
    
    # return a single tensor value
    return _f1score

def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def balanced_loss(y_true, y_pred):
    pt = y_pred * y_true + (1-y_pred) * (1-y_true)
    pt = K.clip(pt, epsilon, 1-epsilon)
    CE = -K.log(pt)
    BL = alpha * CE
    
    return K.sum(BL, axis=1)


def focal_loss(y_true, y_pred):
    pt = y_pred * y_true + (1-y_pred) * (1-y_true)
    pt = K.clip(pt, epsilon, 1-epsilon)
    CE = -K.log(pt)
    FL = alpha * K.pow(1-pt, gamma) * CE
    
    return K.sum(FL, axis=1)


def dice_coef(y_true, y_pred, smooth=0.001):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def cus_loss(y_true, y_pred):
    
    return (1 - beta) * focal_loss(y_true, y_pred) + beta * dice_coef_loss(y_true, y_pred)


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


get_custom_objects().update({
    
    'cus_loss': cus_loss,
    'iou_coef' : iou_coef,
    'f1score' : f1score,
    'precision' : precision,
    'recall' : recall,
    'balanced_loss' : balanced_loss,
    'focal_loss' : focal_loss,
    'dice_coef' : dice_coef,
    'dice_coef_loss' : dice_coef_loss,
    'cus_loss' : cus_loss,
    'tversky_loss' : tversky_loss,
        
})


'''
# ===========================================================
# Data load
# ===========================================================
nodule_num=30
image=np.load('../data/manualgan/%d/241_CTs_resam.npy'%nodule_num,allow_pickle=True)
label=np.load('../data/manualgan/%d/241_labels_resam.npy'%nodule_num,allow_pickle=True)
print(image.shape, label.shape)
print(image[0].shape, label[0].shape)

train_image=image[:-20,:,:,:,:]
train_label=label[:-20,:,:,:,:]

validation_image=image[-20:-10,:,:,:,:]
validation_label=label[-20:-10,:,:,:,:]

test_image=image[-10:,:,:,:,:]
test_label=label[-10:,:,:,:,:]


print('-'*30)
print('tr=',train_image.shape)
print('tr=',train_label.shape)
print('-'*30)
print('val=',validation_image.shape)
print('val=',validation_label.shape)
print('-'*30)
print('te=',test_image.shape)
print('te=',test_label.shape)


# ===========================================================
# Train
# ===========================================================
model = U_Net_3D(128,128,128,1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_delta=1e-8)
earlystopper = EarlyStopping(monitor='val_loss', patience=30, verbose=1)
model_checkpoint = ModelCheckpoint('../result/model/manualgan/supervision_128x128x128_%d_ValRearrange.h5'%nodule_num, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)

callbacks_list = [reduce_lr, model_checkpoint, earlystopper]


losses ={'out_2':dice_coef_loss,
         'out_3':dice_coef_loss,
         'out_4':dice_coef_loss}


# model = multi_gpu_model(model,gpus=8)
model.compile(optimizer=Adam(lr=0.001), loss=losses, metrics=[dice_coef])
print('Fitting model...')
print('-'*200)
hist = model.fit(train_image, [train_label, train_label, train_label], batch_size=1, epochs=200, verbose=1,validation_data= (validation_image, [validation_label, validation_label, validation_label]), shuffle=True, callbacks=callbacks_list)

'''
