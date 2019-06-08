#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 11:57:01 2019

@author: dcase
"""
import numpy as np
import os
def load_pars():
        parameter_dict={}

        parameter_dict['fine_labels'] = ['1-1_small-sounding-engine', '1-2_medium-sounding-engine', 
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
        
        parameter_dict['coarse_labels'] = ['1_engine', '2_machinery-impact', '3_non-machinery-impact', 
                           '4_powered-saw', '5_alert-signal', '6_music', '7_human-voice', '8_dog']
        

        parameter_dict['net_type'] = 'CNN'         #   choice = ['CNN','CRNN','CNN9','CNN_gated']
        parameter_dict['feature_type'] = 'log_mel'        #   choice = ['log_mel','STFT',HPSS']
        parameter_dict['label_level'] = 'coarse'
        parameter_dict['learning_rate'] = 0.001
        parameter_dict['momentum'] = 0.9
        parameter_dict['kernel_size'] = 3   
        parameter_dict['layer_depth'] = [64,128,256,512]          #   [ , , , , ]
        parameter_dict['max_ckpt'] = 20
        parameter_dict['n_epoch'] = 45
        parameter_dict['batch_size'] = 32
        parameter_dict['snapshot'] = 3
        
        parameter_dict['model_path'] = '/home/dcase/c2019/BJS/task5_code/exp_model/test_exp3/CNN5_stft' 
        parameter_dict['train_audio_path'] = '/home/dcase/c2019/dataset/task5/audio/train'
        parameter_dict['val_audio_path'] = '/home/dcase/c2019/dataset/task5/audio/validate'
        parameter_dict['train_label_csv_path'] = '/home/dcase/c2019/dataset/task5/train_'+parameter_dict['label_level']+'_labels_1.csv'
        parameter_dict['val_label_csv_path'] = '/home/dcase/c2019/dataset/task5/val_'+parameter_dict['label_level']+'_labels_1.csv'
        parameter_dict['submission_path'] = os.path.join(parameter_dict['model_path'],'pre_0.csv')
        parameter_dict['annotation_path'] = '/home/dcase/c2019/dataset/task5/annotations.csv'
        parameter_dict['yaml_path'] = '/home/dcase/c2019/dataset/task5/dcase-ust-taxonomy.yaml'
        parameter_dict['noise_path'] = '/home/dcase/c2019/dataset/task5/audio/train_classes/9_noise_presence'
        parameter_dict['goal_path'] = '/home/dcase/c2019/dataset/task5/audio/train_classes'
        
        if not os.path.exists(parameter_dict['model_path']):
            os.makedirs(parameter_dict['model_path'])
        np.save(os.path.join(parameter_dict['model_path'],'parameter_dict.npy'),parameter_dict)
        return parameter_dict
    