# -*- coding: utf-8 -*-
"""
aid_imports
list of each import
---------
@author: maikherbig
"""

import os,shutil,json,re,urllib
import sys,traceback,ast

import numpy as np
import dclab
import h5py,time
import gc,tempfile
import pandas as pd
import tensorflow as tf

from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.client import device_lib

import tensorflow.keras
from tensorflow.keras.models import load_model,Model,Sequential
from tensorflow.keras.layers import add,Add,Concatenate,Input,Dense,Dropout,Flatten,Activation,Conv2D,MaxPooling2D,BatchNormalization,GlobalAveragePooling2D,GlobalMaxPooling2D,concatenate
from tensorflow.keras import backend as K
from keras.models import model_from_json,model_from_config,clone_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
#from keras.utils import np_utils,multi_gpu_model
#from keras.utils.conv_utils import convert_kernel

from tensorflow.keras import regularizers
from tensorflow.keras.utils import get_source_inputs
import keras_applications
from tensorflow import keras

import keras_metrics #side package for precision, recall etc during training
global keras_metrics

import model_zoo 
#from keras2onnx import convert_keras
from onnx import save_model as save_onnx
import tf2onnx


#from mmdnn.conversion._script import convertToIR,IRToCode,convert
#import coremltools
import cv2
import pyqtgraph as pg
from pyqtgraph import Qt
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
#from scipy import ndimage

import aid_start

import io,platform
import copy
from stat import S_IREAD,S_IRGRP,S_IROTH,S_IWRITE,S_IWGRP,S_IWOTH

from tensorboard import program
from tensorboard import default

#from scipy import misc
import sklearn
from sklearn import metrics,preprocessing
from sklearn.utils import _typedefs
import PIL
import openpyxl,xlrd 
import psutil

import aid_img, aid_dl, aid_bin
import aid_frontend
from partial_trainability import partial_trainability

