#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 22:21:15 2019

@author: dcase
"""
import numpy as np
from dcase_functions import *
import os 
import librosa
import pandas as pd


def get_train_audiodata(train_label_csv,audio_path):
    name_list = list(train_label_csv['audio_filename'])
    data = []
    label = np.asarray(train_label_csv)
    label = list(np.asarray(label[:,1:],dtype=np.float32))
    for name in name_list:
        file_path =os.path.join(audio_path,name)
        y,sr = librosa.load(file_path,sr = 22050)
        if len(y)!=sr*10:
            y = np.resize(y,sr*10)
        data.append(y)
    return np.asarray(data,np.float32),np.asarray(label,np.float32)


def get_train_data(train_audio_data,feature):
    train_data=[]
    if feature == 'log_mel':  
        for audio in train_audio_data:
            data = STFT(audio)
            data = convert_to_mel_spectrogram(data)
            data = to_log_mel(data)
            
            train_data.append(data)
    if feature == 'STFT':
        for audio in train_audio_data:
            data = STFT(audio)
            data = power_to_dB(data)
            train_data.append(data)
    if feature == 'HPSS':        
        for audio in train_audio_data:
            data = STFT(audio)
            data_h,data_p = librosa.decompose.hpss(data)
            data_h = power_to_dB(data_h)
            train_data.append(data_h)
    return np.asarray(train_data,np.float32)

def divid_data(data,label):
    divid_data_num = []
    for i in range(data.shape[0]):
        if label[i,0]==0 and label[i,4]==0 and label[i,6]==0 :
            divid_data_num.append(i)
    print(len(divid_data_num))
    divid_data_num = np.asarray(divid_data_num)
    divid_data = data[divid_data_num]
    divid_label = label[divid_data_num]
    return divid_data,divid_label
   

def gen_mel_features(data):
    train_data = []
    for i in range(data.shape[0]):        
        stft_matric = librosa.core.stft(data[i,:],n_fft=1024,hop_length=500,win_length=1024,window='hann')    
        mel_W = librosa.filters.mel(sr=32000,n_fft=1024,n_mels=64,fmin=50,fmax=14000)
        mel_spec = np.dot(mel_W,np.abs(stft_matric)**2)
        log_mel = librosa.core.power_to_db(mel_spec,top_db=None).T
        log_mel = log_mel[:-1]
        train_data.append(log_mel)
    return np.asarray(train_data,dtype=np.float32)


def train_divid_class_label():
    formname = '/home/dcase/c2019/dataset/task5/train_annotations.csv'
    form = pd.read_csv(formname)
    col_name = list(form)
    fine_label = col_name[4:33]
    coarse_label = col_name[62:]
    a = form[coarse_label]
    name = list(form['audio_filename'])
    namelist = []
    labellist = []
    for i in range(0,len(a),3):
        namelist.append(name[i])
        d = np.asarray(a)
        e = d[i:i+3,:]
        e = np.sum(e,axis=0)
        e[e<1]=0
        e[e>=1]=1
        
    
    
