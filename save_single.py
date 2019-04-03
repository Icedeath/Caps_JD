#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 19:02:52 2019

@author: icedeath
"""

#coding=utf

from keras.utils import multi_gpu_model
import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.layers import Lambda
#import matplotlib.pyplot as plt
import tensorflow as tf
from capsulelayers2 import CapsuleLayer, PrimaryCap, Length, Mask
from keras import callbacks
from keras.layers.normalization import BatchNormalization as BN
import argparse
import scipy.io as sio
import h5py
from keras.layers.advanced_activations import ELU
'''
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
'''
K.set_image_data_format('channels_last')


def CapsNet(input_shape, n_class, routings):
    x = layers.Input(shape=input_shape)
    conv1 = layers.Conv2D(filters=64, kernel_size=(1,12), strides=(1,1), padding='same',dilation_rate = 5)(x)
    conv1 = ELU(alpha=0.5)(conv1)
    conv1 = BN()(conv1)
    conv1 = layers.Conv2D(filters=64, kernel_size=(1,12), strides=(1,2), padding='same',dilation_rate = 1)(conv1)
    conv1 = ELU(alpha=0.5)(conv1)
    conv1 = BN()(conv1)
    conv1 = layers.MaxPooling2D((1, 2), strides=(1, 2))(conv1)
    
    conv1 = layers.Conv2D(filters=96, kernel_size=(1,9), strides=1, padding='same',dilation_rate = 4)(conv1)
    conv1 = ELU(alpha=0.5)(conv1)
    conv1 = BN()(conv1)
    conv1 = layers.Conv2D(filters=96, kernel_size=(1,9), strides=1, padding='same',dilation_rate = 4)(conv1)
    conv1 = ELU(alpha=0.5)(conv1)
    conv1 = BN()(conv1)
    conv1 = layers.MaxPooling2D((1, 2), strides=(1, 2))(conv1)
    
    conv1 = layers.Conv2D(filters=128, kernel_size=(1,9), strides=1, padding='same',dilation_rate = 3)(conv1)
    conv1 = ELU(alpha=0.5)(conv1)
    conv1 = BN()(conv1)
    conv1 = layers.Conv2D(filters=128, kernel_size=(1,9), strides=1, padding='same',dilation_rate = 3)(conv1)
    conv1 = ELU(alpha=0.5)(conv1)
    conv1 = BN()(conv1)
    conv1 = layers.MaxPooling2D((1, 2), strides=(1, 2))(conv1)

    conv1 = layers.Conv2D(filters=192, kernel_size=(1,9), strides=1, padding='same',dilation_rate = 2)(conv1)
    conv1 = ELU(alpha=0.5)(conv1)
    conv1 = BN()(conv1)
    conv1 = layers.Conv2D(filters=192, kernel_size=(1,9), strides=1, padding='same',dilation_rate = 2)(conv1)
    conv1 = ELU(alpha=0.5)(conv1)
    conv1 = BN()(conv1)
    conv1 = layers.Conv2D(filters=192, kernel_size=(1,9), strides=1, padding='same',dilation_rate = 2)(conv1)
    conv1 = ELU(alpha=0.5)(conv1)
    conv1 = BN()(conv1)
    conv1 = layers.MaxPooling2D((1, 2), strides=(1, 2))(conv1)

    conv1 = layers.Conv2D(filters=256, kernel_size=(1,6), strides=1, padding='same',dilation_rate = 2)(conv1)
    conv1 = ELU(alpha=0.5)(conv1)
    conv1 = BN()(conv1)
    conv1 = layers.Conv2D(filters=256, kernel_size=(1,6), strides=1, padding='same',dilation_rate = 2)(conv1)
    conv1 = ELU(alpha=0.5)(conv1)
    conv1 = BN()(conv1)
    conv1 = layers.Conv2D(filters=256, kernel_size=(1,6), strides=1, padding='same',dilation_rate = 2)(conv1)
    conv1 = ELU(alpha=0.5)(conv1)
    conv1 = BN()(conv1)
    
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=(1,3),
                             strides=1, padding='same')
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=args.dim_capsule, routings=routings,
                             name='digitcaps')(primarycaps)
    out_caps = Length(name='capsnet')(digitcaps)
    
    

    model = models.Model(x, out_caps)
    return model


def margin_loss(y_true, y_pred, margin = 0.4, threshold = 0.04):
    y_pred = y_pred - 0.5
    t_1 = threshold+0.05
    t_2 = threshold-0.05
    positive_cost = y_true * K.cast(
                    K.less(y_pred, margin), 'float32') * K.pow((y_pred - margin), 2)
    negative_cost = (1 - y_true) * K.cast(
                    K.greater(y_pred, -margin), 'float32') * K.pow((y_pred + margin), 2)
    positive_threshold_cost = y_true * K.cast(
                    K.less(y_pred, t_1), 'float32') * K.pow((y_pred - t_1), 2)
    negative_threshold_cost = (1 - y_true) * K.cast(
                    K.greater(y_pred, -t_2), 'float32') * K.pow((y_pred + -t_2), 2)
    return 0.5 * positive_cost + 0.5 * negative_cost + 0.75*positive_threshold_cost + 0.75*negative_threshold_cost


def margin_loss1(y_true, y_pred, margin = 0.4):
    y_pred = y_pred - 0.5
    positive_cost = y_true * K.cast(
                    K.less(y_pred, margin), 'float32') * K.pow((y_pred - margin), 2)
    negative_cost = (1 - y_true) * K.cast(
                    K.greater(y_pred, -margin), 'float32') * K.pow((y_pred + margin), 2)
    return 0.5 * positive_cost + 0.5 * negative_cost


def train(model, data, args):
    (x_train, y_train) = data

    checkpoint = callbacks.ModelCheckpoint(args.save_file+'-{epoch:02d}.h5', monitor='val_loss', verbose=1, save_best_only=True, 
                                  save_weights_only=True, mode='auto', period=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))
    model = multi_gpu_model(model, gpus=args.num_gpus)
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss= margin_loss,
                  metrics={})
    if args.load == 1:
        model.load_weights(args.save_file)
        print('Loading %s' %args.save_file)
    hist = model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs,
                     validation_split = 0.001, callbacks=[checkpoint, lr_decay])
    return hist.history

def get_accuracy(cm):
    return [float(cm[i,i]/np.sum(cm[0:args.num_classes,i])) for i in xrange(args.num_classes)]


def save_single(args):
    model = CapsNet(input_shape=x_train.shape[1:], n_class=args.num_classes, routings=args.routings)

    p_model = multi_gpu_model(model, gpus=args.num_gpus)
    p_model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss= margin_loss,
                  metrics={})    
    #name = args.save_file.rstrip('.h5') + 'sGPU' + '.h5'
    name = args.save_file + 'sGPU' + '.h5'
    p_model.load_weights(args.save_file)
    model.save_weights(name)
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capsule Network on MAMC.")
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.0003, type=float)
    parser.add_argument('--lr_decay', default=0.988, type=float)
    parser.add_argument('-r', '--routings', default=3, type=int)
    parser.add_argument('-sf', '--save_file', default='5000_Lt_3')
    parser.add_argument('-t', '--test', default=0,type=int)
    parser.add_argument('-l', '--load', default=0,type=int)
    parser.add_argument('-p', '--plot', default=0,type=int)
    parser.add_argument('-n', '--num_classes', default=8)
    parser.add_argument('-dc', '--dim_capsule', default=16)
    parser.add_argument('-tm', '--target_max', default=3, type=int)
    parser.add_argument('-ng', '--num_gpus', default=4, type=int)
    args = parser.parse_args()
    print(args)
    
    K.set_image_data_format('channels_last')
    

    
    x_train = np.ones([10,6000])
    y_train = np.ones([10, 8])
    
    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1], 1)
    
    print('Building model...')
    model = CapsNet(input_shape=x_train.shape[1:], n_class=args.num_classes, routings=args.routings)
    
    print('Saving model to single GPU version...')
    save_single(args)    
    
