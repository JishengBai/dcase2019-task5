#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:29:44 2019

@author: dcase
"""
import tensorflow as tf
from models import CNN_train,CNN9_train,CNN9_gated_train
import numpy as np
import metrics
import os
import pandas as pd
from load_parameters import load_pars
from data_generator import get_train_audiodata,get_train_data
from functions import calculate_loss,get_accuracy,get_batch,get_val_batch,write_pre_csv,shuffle_data
from functions import calculate_scalar_of_tensor,scale
from sklearn.metrics import precision_recall_curve,auc

def train_main(parameter_dict):
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"  
    

    net= parameter_dict['net_type']
    feature = parameter_dict['feature_type']
    train_data_path = parameter_dict['train_data_path']
    val_data_path = parameter_dict['val_data_path']
    train_label_csv_path = parameter_dict['train_label_csv_path']
    val_label_csv_path = parameter_dict['val_label_csv_path']
    lr = parameter_dict['learning_rate']       
    model_path = parameter_dict['model_path']
    max_ckpt = parameter_dict['max_ckpt']
    n_epoch = parameter_dict['n_epoch']
    batch_size = parameter_dict['batch_size']
    snapshot = parameter_dict['snapshot']
    submission_path = parameter_dict['submission_path']
    annotation_path = parameter_dict['annotation_path']
    yaml_path = parameter_dict['yaml_path']
    fine_labels = parameter_dict['fine_labels']
    coarse_labels = parameter_dict['coarse_labels']
    kernel_size = parameter_dict['kernel_size']
    layer_depth = parameter_dict['layer_depth']
    labels_level = parameter_dict['label_level']
        
    label = eval(labels_level+'_labels')
    classes_num = len(label)
    
    ###     net type
    if net=='CNN':
        train_net = CNN_train(kernel_size,layer_depth,classes_num)
    if net=='CNN9':
        train_net = CNN9_train(kernel_size,layer_depth,classes_num)
    if net=='CNN9_gated':
        train_net = CNN9_gated_train(kernel_size,layer_depth,classes_num)
        
        
    ###     load train data  
    train_label_csv = pd.read_csv(train_label_csv_path)  
    train_data = np.load(os.path.join(train_data_path,feature+'.npy')
    train_label = np.asarray(train_label_csv)[:,1:].astype(np.float32)
    print(train_data.shape)  #batch,bin,frame
    frames,bins = train_data[0].shape
    
    ###     load val data
    val_label_csv = pd.read_csv(val_label_csv_path)
    val_label = np.asarray(val_label_csv)[:,1:].astype(np.float32)
    val_namelist = list(val_label_csv['audio_filename'])
    val_data = np.load(val_data_path,feature+'.npy')

    ### calculate mean std
    (mean_train, std_train) = calculate_scalar_of_tensor(np.concatenate(train_data,axis=0))
    
    ###     placeholder
    x=tf.placeholder(tf.float32,shape=[None,frames,bins,1],name='x')
    y=tf.placeholder(tf.float32,shape=[None,classes_num],name='y')
    is_training = tf.placeholder(tf.bool,shape=None,name='is_training')
    
    ###     net output
    output = train_net.forward(input_tensor=x,is_training=is_training)
    loss = calculate_loss(logits=output,labels=y,label_model='n-hot')
    sigmoid = tf.nn.sigmoid(output,name='sigmoid')
    accuracy = get_accuracy(sigmoid=sigmoid,labels=y,label_model='n-hot')
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)            
    learning_rate = tf.Variable(float(lr), trainable=False, dtype=tf.float32)
    learning_rate_decay_op = learning_rate.assign(learning_rate * 0.9)
    with tf.control_dependencies(update_ops):        
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss) 
        
    ###     start session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver(max_to_keep=max_ckpt)
    sess=tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    
    ###     tensorboard summary
    train_summary_dir = os.path.join(model_path, 'summaries', 'train')
    train_summary_writer = tf.summary.FileWriter(train_summary_dir,sess.graph)   
    loss_all=tf.placeholder(tf.float32,shape=None,name='loss_all')
    acc_all=tf.placeholder(tf.float32,shape=None,name='acc_all')
    tf.add_to_collection("loss", loss_all)
    tf.add_to_collection("accuracy", acc_all)
    loss_summary = tf.summary.scalar('loss', loss_all)
    acc_summary = tf.summary.scalar('accuracy', acc_all)
    
    val_summary_dir = os.path.join(model_path, 'summaries', 'val')
    val_micro_auprc_summary_writer = tf.summary.FileWriter(os.path.join(val_summary_dir,'micro_auprc'), sess.graph)
    val_macro_auprc_summary_writer = tf.summary.FileWriter(os.path.join(val_summary_dir,'macro_auprc'), sess.graph)
    val_val_micro_F1score_summary_writer = tf.summary.FileWriter(os.path.join(val_summary_dir,'micro_F1score'), sess.graph)
    val_summary = tf.placeholder(tf.float32,shape=None,name='loss_all')
    tf.add_to_collection("val_summary", val_summary)
    val_summary_op = tf.summary.scalar('val_summary', val_summary)

    ###     train loop
    class_auprc_dict = {}
    mean_auprc_list = []
    for epoch in range(n_epoch):
        train_data,train_label = shuffle_data(train_data,train_label)
        train_label= train_label.reshape(-1,classes_num)
        ###     train
        train_loss = 0 ; train_acc = 0 ; n_batch = 0 
        for train_data_batch,train_label_batch in get_batch(train_data,train_label,batch_size):
            train_data_batch = scale(train_data_batch,mean_train,std_train)
            train_data_batch = train_data_batch.reshape(-1,frames,bins,1)   
            _,train_loss_batch,train_accuracy_batch = sess.run([train_op,loss,accuracy],
               feed_dict={x:train_data_batch,y:train_label_batch,is_training:True})
            train_loss += train_loss_batch ; train_acc += train_accuracy_batch ; n_batch += 1 
        train_loss = train_loss/n_batch;train_acc = train_acc/n_batch
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summaries = sess.run(train_summary_op,feed_dict={loss_all:train_loss,acc_all:train_acc})
        train_summary_writer.add_summary(train_summaries, epoch)
        
        print("step %d" %(epoch))
        print("   train loss: %f" % (train_loss))
        print("   train acc: %f" % (train_acc))
        
        ###     val
        pre=[]
        if ((epoch+1) % snapshot == 0 and epoch > 0) or epoch == n_epoch-1:
            sess.run(learning_rate_decay_op)

            for val_data_batch in get_val_batch(val_data,batch_size):
                val_data_batch = scale(val_data_batch,mean_train,std_train)            
                val_data_batch = val_data_batch.reshape(-1,frames,bins,1) 
                prediction=sess.run(output, feed_dict={x: val_data_batch,is_training:False})
                sigmoid_prediction = sess.run(tf.nn.sigmoid(prediction))
                pre.extend(sigmoid_prediction)
            y_true = np.asarray(val_label,dtype=np.float32)
            y_pre = np.asarray(pre,dtype = np.float32)
            write_pre_csv(val_namelist,pre,labels_level,submission_path,fine_labels,coarse_labels)
            df_dict = metrics.evaluate(prediction_path=submission_path,annotation_path=annotation_path,
                                      yaml_path=yaml_path,mode=labels_level)  
            val_micro_auprc,eval_df = metrics.micro_averaged_auprc(df_dict,return_df=True) 
            val_macro_auprc,class_auprc = metrics.macro_averaged_auprc(df_dict,return_classwise=True)
            thresh_idx_05 = (eval_df['threshold']>=0.5).nonzero()[0][0]
            val_micro_F1score = eval_df['F'][thresh_idx_05]
    
            val_summaries = sess.run(val_summary_op,feed_dict={val_summary:val_micro_auprc})
            val_micro_auprc_summary_writer.add_summary(val_summaries, epoch)
            val_summaries = sess.run(val_summary_op,feed_dict={val_summary:mean_auprc/classes_num})
            val_macro_auprc_summary_writer.add_summary(val_summaries, epoch)
            val_summaries = sess.run(val_summary_op,feed_dict={val_summary:val_micro_F1score})
            val_val_micro_F1score_summary_writer.add_summary(val_summaries, epoch)
            class_auprc_dict['class_auprc_'+str(epoch)] = class_auprc
            np.save(os.path.join(model_path,'class_auprc_dict.npy'),class_auprc_dict)
            print('official')
            print('micro',val_micro_auprc)
            print('micro_F1',val_micro_F1score)
            print('macro',val_macro_auprc)
        
            print('-----save:{}-{}'.format(os.path.join(model_path,'ckeckpoint','model'), epoch))
            saver.save(sess, os.path.join(model_path,'ckeckpoint','model'), global_step=epoch)
            
            mean_auprc_list.append(val_macro_auprc)
            if epoch>40 and mean_auprc_list[-1]<=mean_auprc_list[-2] and mean_auprc_list[-1]<=mean_auprc_list[-3]:
                sess.close()
                os._exit(0)
    
    
        
   
        
        
if __name__=="__main__":
     parameter_dict = load_pars()
     train_main(parameter_dict)
        
        
        
        
        
        
        
