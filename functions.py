#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 09:40:19 2019

@author: dcase
"""
import tensorflow as tf
import numpy as np
import logging


def calculate_loss(logits,labels,label_model):
    if label_model=='one-hot':
        loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels))
    if label_model=='n-hot':
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
    return loss

def get_accuracy(sigmoid,labels,label_model):
    if label_model=='n-hot':
        condition  = tf.less(sigmoid,0.5)
        sigmoid = tf.where(condition,tf.zeros_like(sigmoid),tf.ones_like(sigmoid))
        sigmoid = tf.equal(tf.cast(sigmoid,tf.float32), labels)  
        accuracy = tf.reduce_mean(tf.cast(sigmoid, tf.float32))
    if label_model=='one-hot':
        softmax = tf.nn.softmax(sigmoid)
        correct_prediction = tf.equal(tf.cast(tf.argmax(softmax,1),tf.float32), labels)
        accuracy= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

def shuffle_data(train_data,train_label):
    num = train_data.shape[0]
    arr = np.arange(num)
    np.random.shuffle(arr)
    train_data = train_data[arr]
    train_label = train_label[arr]
    return train_data,train_label
        
def get_batch(train_data,train_label,batch_num):
    for index in range(0,len(train_data),batch_num):       
        if (index+batch_num)<=(len(train_data)):
            excerpt = slice(index, index + batch_num)
        else:
            excerpt = slice(index,len(train_data))
        yield train_data[excerpt], train_label[excerpt]

def get_val_batch(val_data,batch_num):
    for index in range(0,len(val_data),batch_num):       
        if (index+batch_num)<=(len(val_data)):
            excerpt = slice(index, index + batch_num)
        else:
            excerpt = slice(index,len(val_data))
        yield val_data[excerpt]
        
def write_pre_csv(audio_names,outputs,taxonomy_level,submission_path,fine_labels,coarse_labels):
    f = open(submission_path,'w')
    head = ','.join(['audio_filename']+fine_labels+coarse_labels)
    f.write('{}\n'.format(head))
    
    for n,audio_name in enumerate(audio_names):
        if taxonomy_level == 'fine':
            line = ','.join([audio_name]+\
            list(map(str,outputs[n]))+['0.']*len(coarse_labels))
            
        elif taxonomy_level == 'coarse':
            line = ','.join([audio_name]+['0.']*len(fine_labels)+\
            list(map(str,outputs[n])))
        
        else:
            raise Exception('Wrong arg')
        f.write('{}\n'.format(line))
    f.close()
    logging.info('Writing submission to {}'.format(submission_path))
    
def calculate_scalar_of_tensor(x):
    if x.ndim == 2:
        axis = 0
    elif x.ndim == 3:
        axis = (0, 1)

    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)

    return mean, std
  
def scale(x, mean, std):
    return (x - mean) / std
