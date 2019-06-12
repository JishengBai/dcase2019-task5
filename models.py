#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 17:30:39 2019

@author: dcase
"""
import tensorflow as tf



class Conv_block(object):
    def __init__(self,kernel_size,layer_depth):
        
        self.kernel_size = kernel_size
        self.layer_depth = layer_depth
        
    def forward(self,input_tensor,is_training):
        self.input_tensor = input_tensor
        self.is_training = is_training
        conv1 = tf.layers.conv2d(inputs=self.input_tensor,filters=self.layer_depth,
                                 kernel_size=self.kernel_size,strides=1,padding='SAME',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',uniform=True))
        bn1 = tf.layers.batch_normalization(conv1,training=self.is_training)
        relu1 = tf.nn.leaky_relu(bn1)
        
        conv2 = tf.layers.conv2d(inputs=relu1,filters=self.layer_depth,
                                 kernel_size=self.kernel_size,strides=1,padding='SAME',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',uniform=True))
        bn2 = tf.layers.batch_normalization(conv2,training=self.is_training)
        relu2 = tf.nn.leaky_relu(bn2)
       
        return relu2
    
class Conv_gated_block(object):
    def __init__(self,kernel_size,layer_depth):
        
        self.kernel_size = kernel_size
        self.layer_depth = layer_depth
        
    def forward(self,input_tensor,is_training):
        self.input_tensor = input_tensor
        self.is_training = is_training
        conv1 = tf.layers.conv2d(inputs=self.input_tensor,filters=self.layer_depth,
                                 kernel_size=self.kernel_size,strides=1,padding='SAME',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',uniform=True))
        bn1 = tf.layers.batch_normalization(conv1,training=self.is_training)
        relu1 = tf.nn.leaky_relu(bn1)
        sigmoid1 = tf.nn.sigmoid(bn1)
        gated1 = tf.multiply(relu1,sigmoid1)
        
        conv2 = tf.layers.conv2d(inputs=gated1,filters=self.layer_depth,
                                 kernel_size=self.kernel_size,strides=1,padding='SAME',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',uniform=True))
        bn2 = tf.layers.batch_normalization(conv2,training=self.is_training)
        relu2 = tf.nn.leaky_relu(bn2)
        sigmoid2 = tf.nn.sigmoid(bn2)
        gated2 = tf.multiply(relu2,sigmoid2)
       
        return gated2

class CNN_train(object):
    def __init__(self,kernel_size,layer_depth,classes_num):
        
        self.kernel_size = kernel_size
        self.layer_depth = layer_depth
        self.classes_num = classes_num
        
    def forward(self,input_tensor,is_training):
        self.input_tensor = input_tensor
        self.is_training = is_training
        conv1 = tf.layers.conv2d(inputs=input_tensor,filters=self.layer_depth[0],
                                 kernel_size=self.kernel_size,strides=1,padding='SAME',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',uniform=True))
        bn1 = tf.layers.batch_normalization(conv1,training=is_training)
        relu1 = tf.nn.leaky_relu(bn1)
        pool1 = tf.layers.average_pooling2d(relu1,pool_size=2,strides=2,padding='VALID')        
        conv2 = tf.layers.conv2d(inputs=pool1,filters=self.layer_depth[1],
                                 kernel_size=self.kernel_size,strides=1,padding='SAME',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',uniform=True))
        bn2 = tf.layers.batch_normalization(conv2,training=is_training)
        relu2 = tf.nn.leaky_relu(bn2)
        pool2 = tf.layers.average_pooling2d(relu2,pool_size=2,strides=2,padding='VALID')  
        conv3 = tf.layers.conv2d(inputs=pool2,filters=self.layer_depth[2],
                                 kernel_size=self.kernel_size,strides=1,padding='SAME',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',uniform=True))
        bn3 = tf.layers.batch_normalization(conv3,training=is_training)
        relu3 = tf.nn.leaky_relu(bn3)
        pool3 = tf.layers.average_pooling2d(relu3,pool_size=2,strides=2,padding='VALID') 
        conv4 = tf.layers.conv2d(inputs=pool3,filters=self.layer_depth[3],
                                 kernel_size=self.kernel_size,strides=1,padding='SAME',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',uniform=True))
        bn4 = tf.layers.batch_normalization(conv4,training=is_training)
        relu4 = tf.nn.leaky_relu(bn4)
        pool4 = tf.layers.average_pooling2d(relu4,pool_size=1,strides=1,padding='VALID') 
        
        pool4 = tf.reduce_mean(pool4,axis=2)
        pool4 = tf.reduce_max(pool4,axis=1)
        flatten = tf.layers.flatten(pool4)
        output = tf.layers.dense(flatten,units=self.classes_num)
        
        return output


    
class CNN9_train(object):
    def __init__(self,kernel_size,layer_depth,classes_num):
        
        self.kernel_size = kernel_size
        self.layer_depth = layer_depth
        self.classes_num = classes_num
        self.conv_block1 = Conv_block(kernel_size,layer_depth[0])
        self.conv_block2 = Conv_block(kernel_size,layer_depth[1])
        self.conv_block3 = Conv_block(kernel_size,layer_depth[2])
        self.conv_block4 = Conv_block(kernel_size,layer_depth[3])
        
    def forward(self,input_tensor,is_training):
        self.input_tensor = input_tensor
        self.is_training = is_training
        conv1 = self.conv_block1.forward(self.input_tensor,self.is_training)
        pool1 = tf.layers.average_pooling2d(conv1,pool_size=2,strides=2,padding='VALID')        
        conv2 = self.conv_block2.forward(pool1,self.is_training)
        pool2 = tf.layers.average_pooling2d(conv2,pool_size=2,strides=2,padding='VALID')        
        conv3 = self.conv_block3.forward(pool2,self.is_training)
        pool3 = tf.layers.average_pooling2d(conv3,pool_size=2,strides=2,padding='VALID')
        conv4 = self.conv_block4.forward(pool3,self.is_training)
        pool4 = tf.layers.average_pooling2d(conv4,pool_size=1,strides=1,padding='VALID')
        
        pool4 = tf.reduce_mean(pool4,axis=2)
        pool4 = tf.reduce_max(pool4,axis=1)
        flatten = tf.layers.flatten(pool4)
        output = tf.layers.dense(flatten,units=self.classes_num)
        
        return output    

    
class CNN9_gated_train(object):
    def __init__(self,kernel_size,layer_depth,classes_num):
        
        self.kernel_size = kernel_size
        self.layer_depth = layer_depth
        self.classes_num = classes_num
        self.conv_block1 = Conv_gated_block(kernel_size,layer_depth[0])
        self.conv_block2 = Conv_gated_block(kernel_size,layer_depth[1])
        self.conv_block3 = Conv_gated_block(kernel_size,layer_depth[2])
        self.conv_block4 = Conv_gated_block(kernel_size,layer_depth[3])
        
    def forward(self,input_tensor,is_training):
        self.input_tensor = input_tensor
        self.is_training = is_training
        conv1 = self.conv_block1.forward(self.input_tensor,self.is_training)
        pool1 = tf.layers.average_pooling2d(conv1,pool_size=2,strides=2,padding='VALID')        
        conv2 = self.conv_block2.forward(pool1,self.is_training)
        pool2 = tf.layers.average_pooling2d(conv2,pool_size=2,strides=2,padding='VALID')        
        conv3 = self.conv_block3.forward(pool2,self.is_training)
        pool3 = tf.layers.average_pooling2d(conv3,pool_size=2,strides=2,padding='VALID')
        conv4 = self.conv_block4.forward(pool3,self.is_training)
        pool4 = tf.layers.average_pooling2d(conv4,pool_size=1,strides=1,padding='VALID')
        
        pool4 = tf.reduce_mean(pool4,axis=2)
        pool4 = tf.reduce_max(pool4,axis=1)
        flatten = tf.layers.flatten(pool4)
        output = tf.layers.dense(flatten,units=self.classes_num)
        return output    
