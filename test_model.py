#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 23:40:29 2019

@author: dcase
"""

from skimage import io,transform
import tensorflow as tf
import numpy as np
import os
import pandas as pd
import logging
import librosa
import metrics
from functions import calculate_scalar_of_tensor,scale
###---------------------------------------------------
def get_data(label_csv,audio_path):
    name_list = list(label_csv['audio_filename'])
    data = []
    for name in name_list:
        file_path =os.path.join(audio_path,name)
        y,sr = librosa.load(file_path,sr = 32000)
        if len(y)!=sr*10:
            y = np.resize(y,sr*10)
        data.append(y)
    return np.asarray(data,np.float32)

def gen_mel_features(data):
    train_data = []
    for i in range(data.shape[0]):        
        stft_matric = librosa.core.stft(data[i,:],n_fft=1024,hop_length=500,win_length=1024,window='hann')    
        mel_W = librosa.filters.mel(sr=32000,n_fft=1024,n_mels=64,fmin=50,fmax=14000)
        mel_spec = np.dot(mel_W,np.abs(stft_matric)**2)
        log_mel = librosa.core.power_to_db(mel_spec,top_db=None)
        train_data.append(log_mel)
    return np.asarray(train_data,dtype=np.float32)

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
    
###-----------------------------------------------------------
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"   

fine_labels = ['1-1_small-sounding-engine', '1-2_medium-sounding-engine', 
                                '1-3_large-sounding-engine', '1-X_engine-of-uncertain-size', 
                                '2-1_rock-drill', '2-2_jackhammer', '2-3_hoe-ram', '2-4_pile-driver', 
                                '2-X_other-unknown-impact-machinery', '3-1_non-machinery-impact', 
                                '4-1_chainsaw', '4-2_small-medium-rotating-saw', '4-3_large-rotating-saw', 
                                '4-X_other-unknown-powered-saw', '5-1_car-horn', '5-2_car-alarm', 
                                '5-3_siren', '5-4_reverse-beeper', '5-X_other-unknown-alert-signal', 
                                '6-1_stationary-music', '6-2_mobile-music', '6-3_ice-cream-truck', 
                                '6-X_music-from-uncertain-source', '7-1_person-or-small-group-talking', 
                                '7-2_person-or-small-group-shouting', '7-3_large-crowd', 
                                '7-4_amplified-speech', '7-X_other-unknown-human-voice', 
                                '8-1_dog-barking-whining']

coarse_labels = ['1_engine', '2_machinery-impact', '3_non-machinery-impact', 
                   '4_powered-saw', '5_alert-signal', '6_music', '7_human-voice', '8_dog']



model_path = '/home/dcase/c2019/BJS/task5_code/exp_model/test_exp2/CNN9_gated_mfcc/ckeckpoint' #  1
eval_data_path = '/home/dcase/c2019/CC/task5_code/dataset_npy/eval_data/data_32000/mfcc24.npy'  #  2
eval_name_path = '/home/dcase/c2019/CC/task5_code/dataset_npy/eval_data/eval_name.csv'
batch_size = 50
eval_csv_path =  '/home/dcase/c2019/BJS/task5_code/exp_model/test_exp2/test_csv/test1'  #   3
submission_path = eval_csv_path+'/CNN9_gated_mfcc24_pre.csv'    #      4
annotation_path = '/home/dcase/c2019/dataset/task5/annotations.csv'
yaml_path = '/home/dcase/c2019/dataset/task5/dcase-ust-taxonomy.yaml'

eval_csv = pd.read_csv(eval_name_path)
eval_namelist = list(eval_csv['audio_filename'])
eval_data = np.load(eval_data_path)
frames,bins = eval_data[0].shape

train_data_path = '/home/dcase/c2019/CC/task5_code/dataset_npy/data_32000/mfcc/train_mfcc_24.npy'  #  5
val_data_path = '/home/dcase/c2019/CC/task5_code/dataset_npy/data_32000/mfcc/val_mfcc_24.npy'   #    6
train_data = np.load(train_data_path)
val_data = np.load(val_data_path)
all_data = np.concatenate((train_data,val_data),axis=0)
(mean_train, std_train) = calculate_scalar_of_tensor(np.concatenate(all_data,axis=0))
###-----------------------------------
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)
saver = tf.train.import_meta_graph(os.path.join(model_path,'model-41.meta'))    #  7  
#saver.restore(sess,tf.train.latest_checkpoint(model_path))
saver.restore(sess,os.path.join(model_path,'model-41'))
graph = tf.get_default_graph()
x = graph.get_tensor_by_name("x:0")
is_training = graph.get_tensor_by_name("is_training:0")
sigmoid = graph.get_tensor_by_name("sigmoid_8:0")       #     8
pre=[]
for eval_data_batch in get_val_batch(eval_data,batch_size):
        eval_data_batch = scale(eval_data_batch,mean_train,std_train)
        eval_data_batch = eval_data_batch.reshape(-1,frames,bins,1) 
        sigmoid_prediction =sess.run(sigmoid, feed_dict={x: eval_data_batch,is_training:False})
        pre.extend(sigmoid_prediction)
        print(len(pre))
write_pre_csv(eval_namelist,pre,'coarse',submission_path,fine_labels,coarse_labels)     #   9
sess.close()