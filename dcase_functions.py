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



#data preparation
def highpass_butter_filter(N,Wn,data):
    # N the order of the filter
    # digital filters，Wn is`Wn` is normalized from 0 to 1 
    # where 1 is the  Nyquist frequency，Wn = 2*cut off frequency/sample frequency
    # b,a iir filter poles and zeros
    b,a = signal.butter(N,Wn,btype='highpass')     
    filtedData = signal.filtfilt(b,a,data)
    return filtedData

def resample(data,orig_sr,target_sr):
    resample_data = librosa.resample(data,orig_sr, target_sr)
    return resample_data

def extract_audio_chunk(data,jitter_flag,chunk_duration,jitter_duration,sample_rate,drop_chance):   # +jitter_duration
    # data duration must > 2*(chunk_duration+jitter)
    sample_num = len(data)
    data = np.concatenate((data,data),axis=0)

    if jitter_flag == 1:
        chunk_duration = int((chunk_duration+jitter_duration)*sample_rate)
        start_num = random.randint(1,sample_num)     
        drop_num = np.random.random()
        if drop_num>=drop_chance:
            new_data = data[start_num:start_num+chunk_duration]
    #            print('1',len(new_data))
        else:
            cut_start_num = random.randint(start_num,start_num+chunk_duration)
            cut_end_num = random.randint(cut_start_num,cut_start_num+chunk_duration)
            data_before = data[start_num:cut_start_num]
            data_after = data[cut_end_num:cut_end_num+chunk_duration-len(data_before)]
            new_data = np.concatenate((data_before,data_after),axis=0)
    #            print('2',len(new_data),cut_start_num,cut_end_num,len(data_before),len(data_after))

    else:
        chunk_duration = int((chunk_duration-jitter_duration)*sample_rate)
        start_num = random.randint(1,sample_num)
        drop_num = np.random.random()
        if drop_num>=drop_chance:
            new_data = data[start_num:start_num+chunk_duration]
    #            print('3',len(new_data))
        else:
            cut_start_num = random.randint(start_num,start_num+chunk_duration)
            cut_end_num = random.randint(cut_start_num,cut_start_num+chunk_duration)
            data_before = data[start_num:cut_start_num]
            data_after = data[cut_end_num:cut_end_num+chunk_duration-len(data_before)]
            new_data = np.concatenate((data_before,data_after),axis=0)
    #            print('4',len(new_data),cut_start_num,cut_end_num,len(data_before),len(data_after))
    return new_data

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

def add_to_final_data(data,jitter_flag,label0_path,path,add_chance,chunk_duration,
                      jitter_duration,sample_rate,drop_chance):
#    data = amplitudes_random(data,1)    # change
    if np.random.random()<=add_chance[0]:
        add_chunk_0 = add_random_chunk(label0_path)
        add_chunk_0 = extract_audio_chunk(add_chunk_0,jitter_flag,chunk_duration,
                                          jitter_duration,sample_rate,drop_chance)
        add_chunk_0 = amplitudes_random(add_chunk_0,1)
#        print(len(add_chunk_0))
        data = data + add_chunk_0
    if np.random.random()<=add_chance[1]:
        add_chunk_1 = add_random_chunk(label0_path)
        add_chunk_1 = extract_audio_chunk(add_chunk_1,jitter_flag,chunk_duration,
                                          jitter_duration,sample_rate,drop_chance)
        add_chunk_1 = amplitudes_random(add_chunk_1,1)
#        print(len(add_chunk_1))
        data = data + add_chunk_1
    if np.random.random()<=add_chance[2]:
        add_chunk_2 = add_random_chunk(label0_path)
        add_chunk_2 = extract_audio_chunk(add_chunk_2,jitter_flag,chunk_duration,
                                          jitter_duration,sample_rate,drop_chance)
        add_chunk_2 = amplitudes_random(add_chunk_2,1)
#        print(len(add_chunk_2))
        data = data + add_chunk_2
    if np.random.random()<=add_chance[3]:
        add_chunk_3 = add_random_chunk(path)
        add_chunk_3 = extract_audio_chunk(add_chunk_3,jitter_flag,chunk_duration,
                                          jitter_duration,sample_rate,drop_chance)
        add_chunk_3 = amplitudes_random(add_chunk_3,1)
#        print(len(add_chunk_3))
        data = data + add_chunk_3
    if np.random.random()<=add_chance[4]:
        add_chunk_4 = add_random_chunk(path)
        add_chunk_4 = extract_audio_chunk(add_chunk_4,jitter_flag,chunk_duration,
                                          jitter_duration,sample_rate,drop_chance)
        add_chunk_4 = amplitudes_random(add_chunk_4,1)
#        print(len(add_chunk_4))
        data = data + add_chunk_4
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

def frequency_shifting(data):
    random_top_num = np.random.randint(1,10)
    delete_top_num = np.random.choice(9,random_top_num, replace=False)
    data = np.delete(data,delete_top_num,axis=0)
    random_bottom_num = np.random.randint(1,6)
    bottom_choice = np.arange(len(data)-6,len(data),1)
    delete_bottom_num = np.random.choice(bottom_choice,random_bottom_num, replace=False)
    data= np.delete(data,delete_bottom_num,axis=0)
    return data

def convert_to_Image(data):
    data = float_normalization(data)
#    data = RGB_normalization(data)
    data = Image.fromarray(data,mode = 'F')
    return data

def piecewise_time_stretching(data):
    cut_list = []
    width,heigh = data.size
    new_width = 0
    box_width= 0
    while width>=110:
        random_num = random.randint(10,100)
        cut_pic = data.crop([0,0,random_num,heigh])
        resize_width = int(random.randint(90,110)/100*random_num)
        cut_pic = cut_pic.resize([resize_width,heigh],resample = Image.LANCZOS)
        data = data.crop([random_num,0,width,heigh])
        cut_list.append(cut_pic)
        width,heigh = data.size
        continue
    resize_width = int(random.randint(90,110)/100*width)
    data = data.resize([resize_width,heigh],resample = Image.LANCZOS)
    cut_list.append(data)
    for i in range(len(cut_list)):
        width,heigh = cut_list[i].size
        new_width += width
    new_data = Image.new('RGB',[new_width,heigh])
    for i in range(len(cut_list)):
        width,heigh = cut_list[i].size
        box = [box_width,0,box_width+width,heigh]
        box_width+=width
        new_data.paste(cut_list[i],box)
    return new_data

def piecewise_frequency_stretching(data):
    cut_list = []
    width,heigh = data.size
    new_heigh = 0
    box_heigh= 0
    while heigh>=115:
        random_num = random.randint(10,100)
        cut_pic = data.crop([0,0,width,random_num])
        resize_heigh = int(random.randint(95,115)/100*random_num)
        cut_pic = cut_pic.resize([width,resize_heigh],resample = Image.LANCZOS)
        data = data.crop([0,random_num,width,heigh])
        cut_list.append(cut_pic)
        width,heigh = data.size
        continue
    resize_heigh = int(random.randint(90,110)/100*heigh)
    data = data.resize([width,resize_heigh],resample = Image.LANCZOS)
    cut_list.append(data)
    for i in range(len(cut_list)):
        width,heigh = cut_list[i].size
        new_heigh += heigh
    new_data = Image.new('RGB',[width,new_heigh])
    for i in range(len(cut_list)):
        width,heigh = cut_list[i].size
        box = [0,box_heigh,width,box_heigh+heigh]
        box_heigh+=heigh
        new_data.paste(cut_list[i],box)
    return new_data

def resizing(data):
    resizing_chance = np.random.random()
    if resizing_chance>=0.15:
        data = data.resize([299,299],resample = Image.LANCZOS)
    else :
        random_choice = random.choice ( [Image.BICUBIC, Image.NEAREST, Image.BOX, Image.HAMMING,Image.BILINEAR] )
        data = data.resize([299,299],resample = random_choice)
    return data

def duplicate_to_RGB(data):
#    data = float_normalization(data)
    data = RGB_normalization(data)  #!!!
    
    rows, cols = data.shape
    
    color_array = np.empty((rows,cols,3), dtype=np.uint8)
    
    color_array[:,:,0] = data
    
    color_array[:,:,1] = data
    
    color_array[:,:,2] = data

    color_array = Image.fromarray(color_array)
    return color_array

def convert_to_RGB_spectral_int(data):
    colormap_int = np.zeros((256, 3), np.uint8)
    for i in range(0, 255, 1):
       colormap_int[i, 0] = np.int_(np.round(cm.spectral(i)[0] * 255.0))
       colormap_int[i, 1] = np.int_(np.round(cm.spectral(i)[1] * 255.0))
       colormap_int[i, 2] = np.int_(np.round(cm.spectral(i)[2] * 255.0))   
    rows, cols = data.shape
    color_array = np.zeros((rows, cols, 3), np.uint8)
    for i in range(0, rows-1):
        for j in range(0, cols-1):
            x = data[i, j]
            if x>=255:
                x=255
            else:
                color_array[i, j] = colormap_int[x]
    return color_array

def convert_to_RGB_spectral_float(data):
    colormap_float = np.zeros((256, 3), np.float)
    for i in range(0, 255, 1):
       colormap_float[i, 0] = cm.spectral(i)[0]
       colormap_float[i, 1] = cm.spectral(i)[1]
       colormap_float[i, 2] = cm.spectral(i)[2]  
    rows, cols = data.shape
    color_array = np.zeros((rows, cols, 3), np.float32)
    for i in range(0, rows-1):
        for j in range(0, cols-1):
            x = int(data[i, j]*255)
            if x >=255:
                x=255
            else:
                color_array[i, j] = colormap_float[x]
    return color_array

def color_jitter(data):
    return data