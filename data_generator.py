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
    for name in name_list:
        file_path =os.path.join(audio_path,name)
        y,sr = librosa.load(file_path,sr = 32000)
        if len(y)!=sr*10:
            y = np.resize(y,sr*10)
        data.append(y)
    return np.asarray(data,np.float32)


def get_train_data(train_audio_data,feature):
    train_data=[]
    if feature == 'log_mel':  
        for i in range(data.shape[0]):        
            stft_matric = librosa.core.stft(train_audio_data[i,:],n_fft=1024,hop_length=500,win_length=1024,window='hann')    
            mel_W = librosa.filters.mel(sr=32000,n_fft=1024,n_mels=64,fmin=50,fmax=14000)
            mel_spec = np.dot(mel_W,np.abs(stft_matric)**2)
            log_mel = librosa.core.power_to_db(mel_spec,top_db=None).T
            log_mel = log_mel[:-1]
            train_data.append(log_mel)
    if feature == 'STFT':
        for audio in train_audio_data:
            data = STFT(audio)
            data = power_to_dB(data)
            train_data.append(data)
    if feature == 'HPSS_h':        
        for audio in train_audio_data:
            data = STFT(audio)
            data_h,data_p = librosa.decompose.hpss(data)
            data_h = power_to_dB(data_h)
            train_data.append(data_h)
    return np.asarray(train_data,np.float32)

        
 if __name__=='__main__':
    parameter_dict = load_pars()
    train_audio_path = parameter_dict['train_audio_path']
    val_audio_path = parameter_dict['val_audio_path']
    train_data_path = parameter_dict['train_data_path']
    val_data_path = parameter_dict['val_data_path']
    train_label_csv_path = parameter_dict['train_label_csv_path']
    val_label_csv_path = parameter_dict['val_label_csv_path']
    feature = parameter_dict['feature_type']
    
    train_label_csv = pd.read_csv(train_label_csv_path)
    val_label_csv = pd.read_csv(val_label_csv_path)
    train_audio_data = get_train_audiodata(train_label_csv,train_audio_path)
    val_audio_data = get_train_audiodata(val_label_csv,val_audio_path)
    train_data = get_train_data(train_audio_data,feature)
    val_data = get_train_data(val_audio_data,feature)
    np.save(os.path.join(train_data_path,feature+'.npy'),train_data)
    np.save(os.path.join(val_data_path,feature+'.npy'),val_data)
    
    
    
    
    
