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
    

def data_augmentation(data,label,noise_path,class_label_list,goal_path):
    new_data = []
    new_label = []
    data_num = data.shape[0]
        
#    for i in range(data_num):
#        add_noise = 0.5*add_random_chunk(noise_path)
#        add_noise += time_shift(data[i])
#        new_data.append(add_noise)
#        new_label.append(label[i])
    
#    for i in range(data_num):
#        ran_num = np.random.random()
#        if ran_num<=0.60:
#                class_size = 1
#        if 0.60 < ran_num <=0.90:
#                class_size = 2
#        if 0.90 < ran_num :
#                class_size = 3
#        ran_class_id =  np.random.choice([1,3,5,7],class_size)
#        ran_class_choice = []
#        for _id in ran_class_id:
#            ran_class_choice.append(class_label_list[_id]) 
#        new_add_label = np.zeros([8,],dtype=np.float32)
#        for j in range(class_size):
#            add_path = os.path.join(goal_path,ran_class_choice[j])
#            add_data = np.load(add_path+'_presence.npy')
#            ran_start_num = random.randint(0,len(add_data)-32000*10-1)
#            add_data = add_data[ran_start_num:ran_start_num+32000*10]
#        for k in ran_class_id:
#            new_add_label[k]=1.0
#        new_data.append(add_data)
#        new_label.append(new_add_label)
        
    for i in range(data_num):
        ran_num = np.random.random()
        if ran_num<=0.60:
                class_size = 1
        if 0.60 < ran_num <=0.90:
                class_size = 2
        if 0.90 < ran_num :
                class_size = 3
        ran_class_id =  np.random.choice([0,1,3,4,5,6,7],class_size)
        ran_class_choice = []
        for _id in ran_class_id:
            ran_class_choice.append(class_label_list[_id]) 
        noise_data = 0.2*add_random_chunk(noise_path)
        new_add_label = np.zeros([8,],dtype=np.float32)
        for j in range(class_size):
            add_path = os.path.join(goal_path,ran_class_choice[j]+'_presence')
            noise_data += add_random_chunk(add_path)
        for k in ran_class_id:
            new_add_label[k]=1.0
        new_data.append(noise_data)
        new_label.append(new_add_label) 
    return np.asarray(new_data,np.float32),np.asarray(new_label,np.float32)

def get_data(label_csv,audio_path):
    name_list = list(label_csv['audio_filename'])
    data = []
    for name in name_list:
        file_path =os.path.join(audio_path,name)
        y,sr = librosa.load(file_path,sr = 32000)
        if len(y)!=sr*10:
            y = np.concatenate((y, np.zeros(sr*10 - len(y))))
        data.append(y)
    return np.asarray(data,np.float32)

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

def circle_wavs(path):
    file_list = os.listdir(path)
    for i in range(len(file_list)):
        if i==0:
            wav,sr = librosa.load(os.path.join(path,file_list[i]),sr = 32000)
            new_wav = wav
        else:
            wav,sr = librosa.load(os.path.join(path,file_list[i]),sr = 32000)
            new_wav = np.concatenate((new_wav,wav))
    return new_wav


def confusion_mat(labels, predict):
    # labels = [frame, class]
  n_wav, n_class=labels.shape
  conf_mat=np.zeros([n_class, n_class])

  for k in range(n_wav):  # k th frame
    for i in range(n_class): # i th true label
      for j in range(n_class): # j th predict out
        cur_predict_wav=predict[k]
        cur_labels_wav=labels[k]
        if (cur_labels_wav[i]==1 and cur_predict_wav[j]>=0.5 ):
          conf_mat[i][j] += 1
  return conf_mat

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
        labellist.append(e)
    label = np.asarray(labellist)    
    label_sum = np.sum(label,axis=1)   
    label_sum = np.sort(label_sum)    
    label_6 = label[:,5]
    label_8 = label[:,7]        
    label_others = np.delete(label,[5,7],axis=1)   
    label_others_sum = np.sum(label_others,axis=1)    
    label_others_sum[label_others_sum>=1]=1   
    train_3class_label = np.empty((2351,3))   
    train_3class_label[:,0]=label_others_sum  
    train_3class_label[:,1]=label_6   
    train_3class_label[:,2]=label_8   
    class_3_name = ['others_presence',coarse_label[5],coarse_label[7]]   
    train_3class_label = pd.DataFrame(train_3class_label,columns=class_3_name)    
    train_3class_label.insert(0,'audio_filename',np.asarray(namelist))   
    train_3class_label.to_csv('/home/dcase/c2019/dataset/task5/train_3classes_labels.csv',index=0)   
    class_others_name  = [coarse_label[0],coarse_label[1],coarse_label[2],coarse_label[3],coarse_label[4],coarse_label[6]]   
    train_other_labels = pd.DataFrame(label_others,columns = class_others_name)    
    train_other_labels.insert(0,'audio_filename',np.asarray(namelist))   
    train_other_labels.to_csv('/home/dcase/c2019/dataset/task5/train_others_labels.csv',index=0)

if __name__=="__main__":
   
#    train_label_csv_path = '/home/dcase/c2019/dataset/task5/train_coarse_labels_1.csv'
#    train_audio_path = '/home/dcase/c2019/dataset/task5/audio/train'
    val_label_csv_path = '/home/dcase/c2019/dataset/task5/val_coarse_labels_1.csv'
#    val_audio_path = '/home/dcase/c2019/dataset/task5/audio/validate'
#    train_label_csv = pd.read_csv(train_label_csv_path)
#    train_data = get_data(train_label_csv,train_audio_path)
##    train_data = gen_mel_features(train_data)
##    
    val_label_csv = pd.read_csv(val_label_csv_path)
#    val_namelist = list(val_label_csv['audio_filename'])
#    val_data = get_data(val_label_csv,val_audio_path)
#    val_data = gen_mel_features(val_data)
    
#    train_data = np.load('/home/dcase/c2019/dataset/task5/audio/npy_data/train.npy')
#    val_data = np.load('/home/dcase/c2019/dataset/task5/audio/npy_data/val.npy')
#    goal_path = '/home/dcase/c2019/dataset/task5/train_classes_1'
#    coarse_labels = ['1_engine', '2_machinery-impact', '3_non-machinery-impact', 
#                   '4_powered-saw', '5_alert-signal', '6_music', '7_human-voice', '8_dog']
#    noise_path = '/home/dcase/c2019/dataset/task5/train_classes_1/9_noise_presence'
#for i in range(20):
#    aug_data,aug_label = data_augmentation(train_data,goal_path,noise_path,coarse_labels,goal_path)
#    aug_data = gen_mel_features(aug_data)
#    np.save('/home/dcase/c2019/dataset/task5/train_classes_npy/inception_v3_train/aug_data_'+str(i)+'.npy',aug_data)
#    np.save('/home/dcase/c2019/dataset/task5/train_classes_npy/inception_v3_train/aug_label_'+str(i)+'.npy',aug_label)
#    top_path = '/home/dcase/c2019/dataset/task5/train_classes_1'
#    dirnames = os.listdir(top_path)
#    for dirname in dirnames:
#        dir_path = os.path.join(top_path,dirname)
#        new_wav = circle_wavs(dir_path)        
#        new_wav_path = os.path.join(goal_path,dirname)
#        np.save(new_wav_path+'.npy',new_wav)
    
    pre_label_csv_path = '/home/dcase/c2019/BJS/task5_code/exp_model/CNN9_32000/pre_0.csv'
    pre_label_csv = pd.read_csv(pre_label_csv_path)
    val_label = np.asarray(val_label_csv)[:,1:].astype(np.float32)
    pre_label = np.asarray(pre_label_csv)[:,30:].astype(np.float32)
    conf_matric = confusion_mat(val_label,pre_label)
    dog_pre = pre_label[:,7]
    dog_pre_sort = np.argsort(-dog_pre)
    
    