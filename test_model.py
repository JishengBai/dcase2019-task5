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
from functions import get_val_batch,write_pre_csv
    
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



model_path = '~/ckeckpoint' 
eval_data_path = '~/eval_data.npy'  
eval_name_path = '~/eval_name.csv'
batch_size = 50
eval_csv_path =  '~/test_csv'  
submission_path = eval_csv_path+'/pre.csv'
annotation_path = '~/annotations.csv'
yaml_path = '~/dcase-ust-taxonomy.yaml'

eval_csv = pd.read_csv(eval_name_path)
eval_namelist = list(eval_csv['audio_filename'])
eval_data = np.load(eval_data_path)
frames,bins = eval_data[0].shape

train_data_path = '~/log_mel.npy' 
val_data_path = '~/log_mel.npy'   
train_data = np.load(train_data_path)
val_data = np.load(val_data_path)
all_data = np.concatenate((train_data,val_data),axis=0)
(mean_train, std_train) = calculate_scalar_of_tensor(np.concatenate(all_data,axis=0))

###-----------------------------------
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)
saver = tf.train.import_meta_graph(os.path.join(model_path,'model-41.meta'))
saver.restore(sess,tf.train.latest_checkpoint(model_path))

graph = tf.get_default_graph()
x = graph.get_tensor_by_name("x:0")
is_training = graph.get_tensor_by_name("is_training:0")
sigmoid = graph.get_tensor_by_name("sigmoid:0") ## if net==CNN9_gated , "sigmoid_8:0" is replaced
pre=[]
for eval_data_batch in get_val_batch(eval_data,batch_size):
        eval_data_batch = scale(eval_data_batch,mean_train,std_train)
        eval_data_batch = eval_data_batch.reshape(-1,frames,bins,1) 
        sigmoid_prediction =sess.run(sigmoid, feed_dict={x: eval_data_batch,is_training:False})
        pre.extend(sigmoid_prediction)
write_pre_csv(eval_namelist,pre,'coarse',submission_path,fine_labels,coarse_labels)
sess.close()
