# dcase2019-task5: Urban Sound Tagging

DCASE2019 Task5:Urban Sound Tagging is a task to tag audio recordings recorded from urban acoustic sensors. The task includes a fine-grained classification of 23 sound classes and a coarse-grained classification 8 sound classes.

## Functions introduction
data_generator.py:generate train data and val data

load_parameters.py:setup train parameters

train_test_model.py:train model

## Run
**0. Installation** 
Python 3 + tensorflow 1.13 + librosa
**1. download data** 
download development dataset and unzip
**2. generate features**
~/dcase2019-task$ python3 data_generator.py
**3. setup parameters**
revise the parameters in load_parameters.py
**4. train model**
~/dcase2019-task$ python3 train_test_model.py

## Results
