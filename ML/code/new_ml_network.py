#!/home/kiril/python_env_iron_ment/new_proj/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import os
import math

from keras.layers import Input, Dense, Dropout
from keras.layers import experimental, Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, add
from keras.models import Model
from keras.layers.normalization import BatchNormalization

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,KFold
import tensorflow as tf
from keras import backend as K
import numpy as np

import sklearn.metrics as skmetrics

def DeepLayerNN():

    model = Model(input_img, L)
    return model