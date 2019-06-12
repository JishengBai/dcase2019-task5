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
        
def STFT(data):
    data = librosa.core.stft(data,n_fft=1024,hop_length=500,win_length=1024,window='hann')      
    return data

def power_to_dB(data): 
    data = librosa.power_to_db(data,top_db=None)  
    return data
