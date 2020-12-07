# -*- coding: utf-8 -*-
"""
aid_imports
list of each import
---------
@author: maikherbig
"""

import os,shutil,json,re,urllib
import numpy as np
import dclab
import h5py,time
import gc,tempfile
import pandas as pd
import tensorflow as tf
from tensorflow.python.tools import optimize_for_inference_lib
import keras
from keras.models import load_model,Model
from keras.layers import Dense,Activation
from keras import backend as K

import keras_metrics #side package for precision, recall etc during training
global keras_metrics

from keras2onnx import convert_keras
from onnx import save_model as save_onnx
from mmdnn.conversion._script import convertToIR,IRToCode,convert
import coremltools
import cv2
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
import sys,traceback,ast
from pyqtgraph.Qt import QtWidgets
from scipy import ndimage
from tensorflow.python.client import device_lib


import pyqtgraph as pg
from pyqtgraph import Qt

import aid_start

import io,platform
import copy
from stat import S_IREAD,S_IRGRP,S_IROTH,S_IWRITE,S_IWGRP,S_IWOTH

from tensorboard import program
from tensorboard import default

from tensorflow.python.client import device_lib

from scipy import misc
from sklearn import metrics,preprocessing
import PIL
import openpyxl,xlrd 
import psutil

from keras.models import model_from_json,model_from_config,clone_model
from keras.preprocessing.image import load_img
from keras.utils import np_utils,multi_gpu_model
from keras.utils.conv_utils import convert_kernel
import model_zoo 
from keras2onnx import convert_keras
from onnx import save_model as save_onnx

import aid_img, aid_dl, aid_bin
import aid_frontend
from partial_trainability import partial_trainability

from keras.models import Sequential
from keras.layers import Add,Concatenate,Input,Dense,Dropout,Flatten,Activation,Conv2D,MaxPooling2D,BatchNormalization,GlobalAveragePooling2D,GlobalMaxPooling2D,concatenate
from keras.layers.merge import add
from keras import regularizers
from keras.engine.topology import get_source_inputs
import keras_applications
