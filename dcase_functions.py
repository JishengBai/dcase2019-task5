# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 10:45:13 2018

@author: Bai
"""
from scipy import signal
import librosa
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from PIL import Image



def resample(data,orig_sr,target_sr):
    resample_data = librosa.resample(data,orig_sr, target_sr)
    return resample_data

def add_random_chunk(path):
    files = os.listdir(path)
    names = []
    for name in files:
        names.append(name)
    random_num = random.randint(0,len(names)-1)
    data_path = os.path.join(path,names[random_num])
    data,sr = librosa.load(data_path,sr = 32000)
    if len(data)!=sr*10:
            data = np.resize(data,sr*10)
    return data

def amplitudes_random(data,chance):
    random_num = np.random.random()
    if random_num >= (1-chance):
        factor = np.random.random()
        data = data*factor
    return data

def time_shift(data):
    time_length = len(data)
    cut_num = random.randint(0,time_length)
    data_end = data[cut_num:]
    data_beginning = data[:cut_num]
    data = np.concatenate((data_end,data_beginning),axis=0)
    return data
        
def STFT(data):
    data = librosa.core.stft(data,n_fft=1024,hop_length=500,win_length=1024,window='hann')      
    return data

def STFT_CRNN(data):
    data = librosa.core.stft(data,n_fft=1024,hop_length=512,win_length=1024,window='hann')      
    return data

def float_normalization(data):
    datamax,datamin = data.max(),data.min()
    data = (data-datamin)/(datamax-datamin)
    data = data.astype('float32')
    return data

def RGB_normalization(data):
    datamax,datamin = data.max(),data.min()
    data = 255*(data-datamin)/(datamax-datamin)
    data = data.astype('int32')
    return data

def power_to_dB(data):
    data = np.abs(data)
    data = float_normalization(data)    

    data = librosa.power_to_db(data,top_db=None)
    
    return data

def to_log_mel(mel_spec):
    log_mel = librosa.core.power_to_db(mel_spec,top_db=None)
    return  log_mel

def convert_to_mel_spectrogram(data):
    mel_W = librosa.filters.mel(sr=22050,n_fft=1024,n_mels=64,fmin=50,fmax=11025)
    mel_spec = np.dot(mel_W,np.abs(data)**2)
    return mel_spec

def convert_to_mel_spectrogram_CRNN(data):
    data = librosa.feature.melspectrogram(S=data,n_mels=80,fmin=50,fmax=10300)
    return data
