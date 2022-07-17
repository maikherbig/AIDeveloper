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
import h5py,time
import hdf5plugin
import gc,tempfile
import pandas as pd
import tensorflow as tf
from tensorflow import *
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.client import device_lib

import tensorflow.keras
from tensorflow.keras.models import load_model,Model,Sequential
from tensorflow.keras.layers import add,Add,Concatenate,Input,Dense,Dropout,Flatten,Activation,Conv2D,MaxPooling2D,BatchNormalization,GlobalAveragePooling2D,GlobalMaxPooling2D,concatenate
from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_json,model_from_config,clone_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from tensorflow import lite
from tensorflow.lite import *
from tensorflow._api.v2.compat.v1.compat.v2.keras import *

import tensorflow._api.v2.compat.v2.compat.v2.keras.experimental
import tensorflow._api.v2.compat.v2.keras.utils.experimental
import tensorflow._api.v2.compat.v2.compat.v2.keras.applications.vgg16
import tensorflow._api.v2.compat.v2.compat.v1.keras.applications.mobilenet_v2
import tensorflow._api.v2.compat.v1.keras.activations
import tensorflow._api.v2.compat.v2.compat.v1.keras.applications.inception_resnet_v2
import tensorflow._api.v2.compat.v2.compat.v2.keras.datasets.mnist
import tensorflow._api.v2.compat.v1.keras.optimizers
import tensorflow._api.v2.compat.v2.compat.v2.keras.applications.imagenet_utils
import tensorflow._api.v2.compat.v2.compat.v2.keras.losses
import tensorflow._api.v2.compat.v1.compat.v1.keras.layers.experimental.preprocessing
import tensorflow._api.v2.compat.v1.compat.v1.keras.applications.resnet50
import tensorflow._api.v2.compat.v2.compat.v2.keras.applications.mobilenet
import tensorflow._api.v2.compat.v2.compat.v1.keras.optimizers
#import tensorflow._api.v2.compat.v1.compat.v1.rnn_cell
import tensorflow._api.v2.compat.v1.compat.v1.keras.optimizers
# import tensorflow._api.v2.compat.v2.compat.v1.layers
import tensorflow._api.v2.compat.v2.compat.v1.keras.applications.xception
# import tensorflow._api.v2.compat.v2.compat.v2.keras.__internal__.backend
import tensorflow._api.v2.compat.v1.estimator.experimental
import tensorflow._api.v2.compat.v2.compat.v2.keras.preprocessing.text
import tensorflow._api.v2.compat.v1.compat.v2.keras.preprocessing.text
import tensorflow._api.v2.compat.v2.compat.v1.keras.callbacks
import tensorflow._api.v2.compat.v1.compat.v1.keras.estimator
import tensorflow._api.v2.compat.v2.keras.applications.densenet
import tensorflow._api.v2.compat.v2.compat.v2.keras.backend
import tensorflow._api.v2.compat.v1.compat.v1.keras.applications.mobilenet_v3
import tensorflow._api.v2.compat.v1.compat.v1.keras.layers.experimental
import tensorflow._api.v2.compat.v2.compat.v2.keras.applications.mobilenet_v3
# import tensorflow._api.v2.compat.v2.compat.v2.keras.__internal__.utils
import tensorflow._api.v2.compat.v1.compat.v1.keras.callbacks.experimental
import tensorflow._api.v2.compat.v2.keras.wrappers
import tensorflow._api.v2.compat.v2.compat.v1.keras.applications.resnet50
import tensorflow._api.v2.compat.v2.compat.v1.keras.applications.imagenet_utils
# import tensorflow._api.v2.compat.v1.compat.v2.keras.__internal__.models
import tensorflow._api.v2.compat.v2.keras.applications.resnet
import tensorflow._api.v2.compat.v2.keras.optimizers.schedules
import tensorflow._api.v2.compat.v1.compat.v2.keras.datasets.boston_housing
# import tensorflow._api.v2.compat.v1.compat.v2.keras.__internal__
import tensorflow._api.v2.compat.v1.keras.applications.vgg19
# import tensorflow._api.v2.compat.v1.compat.v1.keras.__internal__.legacy.layers.experimental
import tensorflow._api.v2.compat.v2.compat.v1.keras.datasets
import tensorflow._api.v2.compat.v2.keras.layers.experimental.preprocessing
import tensorflow._api.v2.compat.v2.keras.models
import tensorflow._api.v2.compat.v1.compat.v1.keras.callbacks
# import tensorflow._api.v2.compat.v1.compat.v1.keras.__internal__.legacy.rnn_cell
import tensorflow._api.v2.compat.v1.compat.v2.keras.applications.nasnet
import tensorflow._api.v2.compat.v1.compat.v1.keras.datasets
import tensorflow._api.v2.compat.v2.compat.v2.keras.metrics
import tensorflow._api.v2.compat.v1.keras
import tensorflow._api.v2.compat.v2.compat.v2.keras.preprocessing.sequence
import tensorflow._api.v2.compat.v1.compat.v2.keras.wrappers
import tensorflow._api.v2.compat.v2.compat.v2.keras.utils.experimental
import tensorflow._api.v2.compat.v2.compat.v2.estimator.experimental
import tensorflow._api.v2.compat.v1.compat.v2.keras.regularizers
# import tensorflow._api.v2.compat.v1.compat.v1.layers.experimental
import tensorflow._api.v2.compat.v1.compat.v1.keras.applications.resnet
import tensorflow._api.v2.compat.v2.compat.v2.keras.optimizers.schedules
import tensorflow._api.v2.compat.v2.keras.optimizers
# import tensorflow._api.v2.compat.v2.keras.__internal__.losses
import tensorflow._api.v2.compat.v2.compat.v2.keras.datasets.fashion_mnist
import tensorflow._api.v2.compat.v2.compat.v1.keras.applications.resnet_v2
import tensorflow._api.v2.compat.v1.compat.v1.keras.preprocessing
import tensorflow._api.v2.compat.v2.compat.v2.keras.preprocessing.image
import tensorflow._api.v2.compat.v1.keras.applications.mobilenet_v3
import tensorflow._api.v2.compat.v1.compat.v1.estimator.export
import tensorflow._api.v2.compat.v2.keras.datasets.cifar10
import tensorflow._api.v2.compat.v2.compat.v2.keras.mixed_precision.experimental
import tensorflow._api.v2.compat.v2.keras.applications.xception
import tensorflow._api.v2.compat.v1.compat.v1.keras.applications.resnet_v2
import tensorflow._api.v2.compat.v1.compat.v2.keras.callbacks.experimental
import tensorflow._api.v2.compat.v1.keras.preprocessing
# import tensorflow._api.v2.compat.v2.compat.v2.keras.__internal__
import tensorflow._api.v2.compat.v1.keras.datasets.mnist
import tensorflow._api.v2.compat.v2.keras.utils
import tensorflow._api.v2.compat.v1.keras.experimental
import tensorflow._api.v2.compat.v1.keras.applications.resnet50
import tensorflow._api.v2.compat.v1.compat.v2.keras.datasets.fashion_mnist
import tensorflow._api.v2.compat.v2.compat.v1.estimator.experimental
import tensorflow._api.v2.compat.v1.compat.v2.keras.applications
import tensorflow._api.v2.compat.v1.keras.applications.nasnet
import tensorflow._api.v2.compat.v2.compat.v1.keras.datasets.mnist
import tensorflow._api.v2.compat.v1.compat.v2.keras.datasets.cifar10
# import tensorflow._api.v2.compat.v1.keras.__internal__.legacy.layers.experimental
import tensorflow._api.v2.compat.v2.keras.applications.mobilenet_v2
import tensorflow._api.v2.compat.v2.keras.mixed_precision.experimental
import tensorflow._api.v2.compat.v2.keras.applications.imagenet_utils
import tensorflow._api.v2.compat.v2.keras.applications.vgg19
# import tensorflow._api.v2.compat.v1.compat.v2.keras.__internal__.backend
import tensorflow._api.v2.compat.v2.compat.v2.keras.applications.mobilenet_v2
import tensorflow._api.v2.compat.v1.compat.v2.keras.applications.imagenet_utils
import tensorflow._api.v2.compat.v2.compat.v2.keras.applications.xception
import tensorflow._api.v2.compat.v2.compat.v2.keras.activations
import tensorflow._api.v2.compat.v1.compat.v2.keras
# import tensorflow._api.v2.compat.v2.compat.v1.keras.__internal__.legacy.layers.experimental
import tensorflow._api.v2.compat.v1.compat.v1.estimator.tpu.experimental
import tensorflow._api.v2.compat.v1.compat.v1.keras.optimizers.schedules
import tensorflow._api.v2.compat.v2.keras.initializers
import tensorflow._api.v2.compat.v2.keras.mixed_precision
import tensorflow._api.v2.compat.v2.compat.v1.keras.preprocessing.sequence
import tensorflow._api.v2.compat.v2.keras.applications.nasnet
import tensorflow._api.v2.compat.v1.keras.datasets.imdb
###import tensorflow._api.v2.compat.v2.estimator.inputs
import tensorflow._api.v2.compat.v1.compat.v2.keras.datasets.reuters
import tensorflow._api.v2.compat.v2.compat.v1.estimator.export
import tensorflow._api.v2.compat.v1.keras.applications.inception_resnet_v2
import tensorflow._api.v2.compat.v1.compat.v1.keras.datasets.reuters
import tensorflow._api.v2.compat.v2.compat.v2.keras
import tensorflow._api.v2.compat.v2.keras.losses
import tensorflow._api.v2.compat.v2.compat.v1.keras.applications.vgg19
# import tensorflow._api.v2.compat.v2.compat.v2.keras.__internal__.models
import tensorflow._api.v2.compat.v1.compat.v1.keras.constraints
import tensorflow._api.v2.compat.v1.compat.v2.keras.applications.efficientnet
# import tensorflow._api.v2.compat.v2.compat.v1.keras.__internal__.legacy.rnn_cell
import tensorflow._api.v2.compat.v1.keras.datasets.fashion_mnist
import tensorflow._api.v2.compat.v1.keras.mixed_precision
import tensorflow._api.v2.compat.v2.compat.v2.keras.applications.resnet
import tensorflow._api.v2.compat.v1.compat.v1.keras.datasets.mnist
import tensorflow._api.v2.compat.v1.compat.v1.keras.utils
import tensorflow._api.v2.compat.v2.compat.v1.keras.layers
import tensorflow._api.v2.compat.v2.compat.v1.keras.callbacks.experimental
import tensorflow._api.v2.compat.v1.compat.v1.keras
# import tensorflow._api.v2.compat.v1.compat.v1.layers
import tensorflow._api.v2.compat.v1.compat.v2.keras.applications.mobilenet
import tensorflow._api.v2.compat.v2.compat.v2.keras.applications.resnet_v2
import tensorflow._api.v2.compat.v2.compat.v1.keras.mixed_precision
import tensorflow._api.v2.compat.v2.keras
# import tensorflow._api.v2.compat.v2.keras.__internal__.backend
import tensorflow._api.v2.compat.v2.keras.estimator
import tensorflow._api.v2.compat.v2.keras.activations
import tensorflow._api.v2.compat.v2.compat.v2.keras.optimizers
import tensorflow._api.v2.compat.v2.compat.v2.keras.applications.vgg19
import tensorflow._api.v2.compat.v1.compat.v1.estimator
import tensorflow._api.v2.compat.v1.compat.v2.keras.initializers
import tensorflow._api.v2.compat.v1.compat.v1.keras.datasets.fashion_mnist
import tensorflow._api.v2.compat.v1.keras.applications.vgg16
import tensorflow._api.v2.compat.v1.compat.v2.keras.datasets
import tensorflow._api.v2.compat.v1.keras.applications.resnet_v2
import tensorflow._api.v2.compat.v2.compat.v1.keras.preprocessing.text
import tensorflow._api.v2.compat.v2.compat.v1.keras.utils
import tensorflow._api.v2.compat.v1.compat.v2.keras.estimator
import tensorflow._api.v2.compat.v1.compat.v2.estimator.export
import tensorflow._api.v2.compat.v1.compat.v2.keras.layers
import tensorflow._api.v2.compat.v1.compat.v2.keras.constraints
import tensorflow._api.v2.compat.v1.compat.v2.estimator
import tensorflow._api.v2.compat.v2.compat.v1.keras.optimizers.schedules
import tensorflow._api.v2.compat.v2.compat.v2.keras.applications.resnet50
import tensorflow._api.v2.compat.v1.compat.v1.keras.applications.densenet
import tensorflow._api.v2.compat.v2.compat.v1.keras.datasets.imdb
import tensorflow._api.v2.compat.v1.compat.v1.keras.backend
import tensorflow._api.v2.compat.v2.compat.v1.keras.layers.experimental.preprocessing
import tensorflow._api.v2.compat.v1.compat.v2.keras.preprocessing.sequence
# import tensorflow._api.v2.compat.v1.compat.v1.rnn_cell
import tensorflow._api.v2.compat.v1.compat.v1.keras.applications.vgg16
import tensorflow._api.v2.compat.v2.keras.applications.mobilenet
import tensorflow._api.v2.compat.v1.compat.v2.keras.applications.inception_resnet_v2
import tensorflow._api.v2.compat.v2.compat.v1.keras.layers.experimental
import tensorflow._api.v2.compat.v2.estimator
# import tensorflow._api.v2.compat.v1.nn.layers
import tensorflow._api.v2.compat.v2.compat.v1.keras.wrappers.scikit_learn
import tensorflow._api.v2.compat.v2.compat.v2.keras.models
import tensorflow._api.v2.compat.v1.keras.datasets.reuters
import tensorflow._api.v2.compat.v1.compat.v2.keras.datasets.mnist
# import tensorflow._api.v2.compat.v1.compat.v1.keras.__internal__.legacy.layers
import tensorflow._api.v2.compat.v1.compat.v1.keras.activations
# import tensorflow._api.v2.compat.v1.keras.__internal__.legacy.layers
import tensorflow._api.v2.compat.v2.keras.preprocessing
import tensorflow._api.v2.compat.v2.compat.v1.keras.datasets.cifar10
# import tensorflow._api.v2.compat.v2.keras.__internal__.utils
# import tensorflow._api.v2.compat.v1.compat.v2.keras.__internal__.utils
import tensorflow._api.v2.compat.v2.compat.v2.keras.datasets.cifar100
import tensorflow._api.v2.compat.v1.compat.v1.keras.applications.mobilenet_v2
import tensorflow._api.v2.compat.v1.keras.backend
import tensorflow._api.v2.compat.v1.keras.callbacks.experimental
import tensorflow._api.v2.compat.v1.keras.applications.mobilenet
import tensorflow._api.v2.compat.v2.compat.v2.keras.datasets.cifar10
import tensorflow._api.v2.compat.v1.compat.v2.summary
import tensorflow._api.v2.compat.v1.compat.v2.keras.mixed_precision.experimental
import tensorflow._api.v2.compat.v2.keras.applications.resnet_v2
import tensorflow._api.v2.compat.v1.compat.v2.keras.applications.mobilenet_v2
import tensorflow._api.v2.compat.v1.estimator.tpu.experimental
import tensorflow._api.v2.compat.v2.compat.v1.keras.datasets.boston_housing
import tensorflow._api.v2.compat.v1.keras.applications.inception_v3
import tensorflow._api.v2.compat.v1.compat.v2.keras.datasets.imdb
import tensorflow._api.v2.compat.v2.compat.v1.keras.preprocessing
import tensorflow._api.v2.compat.v2.compat.v1.keras.experimental
import tensorflow._api.v2.compat.v2.keras.backend
import tensorflow._api.v2.compat.v2.compat.v2.estimator.export
import tensorflow._api.v2.compat.v1.compat.v2.keras.applications.resnet50
import tensorflow._api.v2.compat.v1.compat.v1.keras.applications.nasnet
import tensorflow._api.v2.compat.v2.compat.v2.keras.layers.experimental
import tensorflow._api.v2.compat.v2.keras.applications.mobilenet_v3
import tensorflow._api.v2.compat.v1.keras.datasets
import tensorflow._api.v2.compat.v2.compat.v2.keras.wrappers.scikit_learn
import tensorflow._api.v2.compat.v1.keras.regularizers
import tensorflow._api.v2.compat.v1.compat.v1.keras.applications.vgg19
import tensorflow._api.v2.compat.v1.compat.v1.keras.premade
import tensorflow._api.v2.compat.v1.keras.datasets.cifar10
import tensorflow._api.v2.compat.v1.compat.v2.keras.applications.vgg16
##import tensorflow._api.v2.compat.v2.v2
import tensorflow._api.v2.compat.v1.compat.v1.keras.mixed_precision.experimental
import tensorflow._api.v2.compat.v2.compat.v1.keras.datasets.fashion_mnist
# import tensorflow._api.v2.compat.v2.compat.v1.layers.experimental
import tensorflow._api.v2.compat.v2.keras.premade
import tensorflow._api.v2.compat.v2.compat.v1.keras.datasets.reuters
import tensorflow._api.v2.compat.v1.compat.v1.keras.datasets.imdb
import tensorflow._api.v2.compat.v1.compat.v2.keras.premade
import tensorflow._api.v2.compat.v1.compat.v1.keras.metrics
import tensorflow._api.v2.compat.v1.keras.initializers
import tensorflow._api.v2.compat.v1.keras.wrappers
import tensorflow._api.v2.compat.v1.keras.preprocessing.text
import tensorflow._api.v2.compat.v2.compat.v1.keras.wrappers
import tensorflow._api.v2.compat.v2.compat.v2.keras.datasets.reuters
import tensorflow._api.v2.compat.v1.compat.v2.keras.applications.resnet
import tensorflow._api.v2.compat.v1.compat.v2.keras.applications.inception_v3
import tensorflow._api.v2.compat.v2.keras.datasets.mnist
import tensorflow._api.v2.compat.v2.keras.datasets
# import tensorflow._api.v2.compat.v1.keras.__internal__.legacy.rnn_cell
import tensorflow._api.v2.compat.v1.compat.v1.keras.preprocessing.sequence
import tensorflow._api.v2.compat.v2.compat.v2.keras.mixed_precision
import tensorflow._api.v2.compat.v1.compat.v1.keras.applications
##import tensorflow._api.v2.compat.v2.compat.v2.v2
import tensorflow._api.v2.compat.v2.keras.callbacks
import tensorflow._api.v2.compat.v2.compat.v1.keras.applications.efficientnet
import tensorflow._api.v2.compat.v1.compat.v1.keras.applications.xception
import tensorflow._api.v2.compat.v2.estimator.experimental


#from keras.utils import np_utils,multi_gpu_model
#from keras.utils.conv_utils import convert_kernel

from tensorflow.keras import regularizers
from tensorflow.keras.utils import get_source_inputs
import keras_applications
from tensorflow import keras

# import keras_metrics #side package for precision, recall etc during training
# global keras_metrics

import model_zoo 
#from keras2onnx import convert_keras
from onnx import save_model as save_onnx
import tf2onnx


#from mmdnn.conversion._script import convertToIR,IRToCode,convert
#import coremltools
import cv2
from cv2 import *
import pyqtgraph as pg
from pyqtgraph import *
from pyqtgraph.graphicsItems.ViewBox import axisCtrlTemplate_pyqt5
from pyqtgraph.graphicsItems.PlotItem import plotConfigTemplate_pyqt5
from pyqtgraph.graphicsItems import *
from pyqtgraph import Qt
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
from pyqtgraph.imageview import ImageViewTemplate_pyqt5
from pyqtgraph.imageview import *
#from scipy import ndimage

import aid_start

import io,platform
import copy
from stat import S_IREAD,S_IRGRP,S_IROTH,S_IWRITE,S_IWGRP,S_IWOTH

# from tensorboard import program
# from tensorboard import default

from scipy import *
from scipy import stats
from scipy.spatial.transform import _rotation_groups
from scipy.spatial.transform import *
from scipy.special import cython_special
from scipy.special import *
import sklearn
from sklearn import metrics,preprocessing
from sklearn.utils import _typedefs
import PIL
import openpyxl,xlrd 
import psutil

import aid_img, aid_dl, aid_bin
import aid_frontend
from partial_trainability import partial_trainability

