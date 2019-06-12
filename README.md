# dcase2019-task5: Urban Sound Tagging

DCASE2019 Task5:Urban Sound Tagging is a task to tag audio recordings recorded from urban acoustic sensors. The task includes a fine-grained classification of 23 sound classes and a coarse-grained classification 8 sound classes.

## Labels

The train labels and val labels are summarised with OR function

## Train Run 

**0. Installation** 

Python 3 + tensorflow 1.13 + librosa

**1. download data** 

download development dataset and unzip

**2. setup parameters**

revise the parameters in load_parameters.py

**3. generate features**

~/dcase2019-task$ python3 data_generator.py

**4. train model**

~/dcase2019-task$ python3 train_model.py

## Eval Run

revise the parameters in test_model.py

~/dcase2019-task$ python3 test_model.py
