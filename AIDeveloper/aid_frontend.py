import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
import sys,os,json,traceback,ast
import numpy as np
import cv2
import model_zoo 
import aid_start,aid_dl,aid_bin #import a module that sits in the AIDeveloper folder
dir_root = os.path.dirname(aid_start.__file__)#ask the module for its origin
dir_settings = os.path.join(dir_root,"aid_settings.json")#dir to settings
with open(dir_settings) as f:
    Default_dict = json.load(f)
    #Older versions of AIDeveloper might not have the Icon theme option->add it!
    if "Icon theme" not in Default_dict.keys():
        Default_dict["Icon theme"] = "Icon theme 1"
    if "Path of last model" not in Default_dict.keys():
        Default_dict["Path of last model"] = 'c:\\'

tooltips = aid_start.get_tooltips()

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtWidgets.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtWidgets.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtWidgets.QApplication.translate(context, text, disambig)


class MyTable(QtWidgets.QTableWidget):
    dropped = QtCore.pyqtSignal(list)

    def __init__(self,  rows, columns, parent):
        super(MyTable, self).__init__(rows, columns, parent)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        #self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.drag_item = None
        self.drag_row = None

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        self.drag_item = None
        if event.mimeData().hasUrls:
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
            links = []
            for url in event.mimeData().urls():
                links.append(str(url.toLocalFile()))
            self.dropped.emit(links)
        else:
            event.ignore()       
        
    def startDrag(self, supportedActions):
        super(MyTable, self).startDrag(supportedActions)
        self.drag_item = self.currentItem()
        self.drag_row = self.row(self.drag_item)


def MyExceptionHook(etype, value, trace):
    """
    Copied from: https://github.com/ZELLMECHANIK-DRESDEN/ShapeOut/blob/07d741db3bb5685790d9f9f6df394cd9577e8236/shapeout/gui/frontend.py
    Handler for all unhandled exceptions.
 
    :param `etype`: the exception type (`SyntaxError`, `ZeroDivisionError`, etc...);
    :type `etype`: `Exception`
    :param string `value`: the exception error message;
    :param string `trace`: the traceback header, if any (otherwise, it prints the
     standard Python header: ``Traceback (most recent call last)``.
    """
    tmp = traceback.format_exception(etype, value, trace)
    exception = "".join(tmp)
    msg = QtWidgets.QMessageBox()
    msg.setIcon(QtWidgets.QMessageBox.Information)       
    msg.setText(exception)
    msg.setWindowTitle("Error")
    msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
    msg.exec_()
    return

class WorkerSignals(QtCore.QObject):
    '''
    Code inspired from here: https://www.learnpyqt.com/courses/concurrent-execution/multithreading-pyqt-applications-qthreadpool/
    
    Defines the signals available from a running worker thread.
    Supported signals are:
    finished
        No data
    error
        `tuple` (exctype, value, traceback.format_exc() )
    result
        `object` data returned from processing, anything
    progress
        `int` indicating % progress
    history
        `dict` containing keras model history.history resulting from .fit
    '''
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(tuple)
    result = QtCore.pyqtSignal(object)
    progress = QtCore.pyqtSignal(int)
    history = QtCore.pyqtSignal(dict)

class Worker(QtCore.QRunnable):
    '''
    Code inspired/copied from: https://www.learnpyqt.com/courses/concurrent-execution/multithreading-pyqt-applications-qthreadpool/
    Worker thread
    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.
    :param callback: The function callback to run on this worker thread. Supplied args and 
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function
    '''
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress
        self.kwargs['history_callback'] = self.signals.history

    @QtCore.pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done





def setup_main_ui(self,gpu_nr):
    self.setObjectName(_fromUtf8("MainWindow"))
    self.resize(900, 600)

    sys.excepthook = MyExceptionHook
    self.centralwidget = QtWidgets.QWidget(self)
    self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
    self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
    self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
    self.tabWidget_Modelbuilder = QtWidgets.QTabWidget(self.centralwidget)
    self.tabWidget_Modelbuilder.setObjectName(_fromUtf8("tabWidget_Modelbuilder"))
    self.tab_Build = QtWidgets.QWidget()
    self.tab_Build.setObjectName(_fromUtf8("tab_Build"))
    self.gridLayout_17 = QtWidgets.QGridLayout(self.tab_Build)
    self.gridLayout_17.setObjectName(_fromUtf8("gridLayout_17"))
    self.splitter_5 = QtWidgets.QSplitter(self.tab_Build)
    self.splitter_5.setOrientation(QtCore.Qt.Vertical)
    self.splitter_5.setObjectName(_fromUtf8("splitter_5"))
    self.splitter_3 = QtWidgets.QSplitter(self.splitter_5)
    self.splitter_3.setOrientation(QtCore.Qt.Vertical)
    self.splitter_3.setObjectName(_fromUtf8("splitter_3"))
    self.groupBox_dragdrop = QtWidgets.QGroupBox(self.splitter_3)
    self.groupBox_dragdrop.setMinimumSize(QtCore.QSize(0, 200))
    self.groupBox_dragdrop.setObjectName(_fromUtf8("groupBox_dragdrop"))
    self.gridLayout_8 = QtWidgets.QGridLayout(self.groupBox_dragdrop)
    self.gridLayout_8.setObjectName(_fromUtf8("gridLayout_8"))
    
    self.table_dragdrop = MyTable(0,11,self.groupBox_dragdrop) #table with 9 columns
    self.table_dragdrop.setObjectName(_fromUtf8("table_dragdrop"))
    header_labels = ["File", "Class" ,"T", "V", "Show","Events total","Events/Epoch","PIX","Shuffle","Zoom","Xtra_In"]

    self.table_dragdrop.setHorizontalHeaderLabels(header_labels) 
    header = self.table_dragdrop.horizontalHeader()
    for i in [1,2,3,4,5,6,7,8,9,10]:#range(len(header_labels)):
        header.setResizeMode(i, QtWidgets.QHeaderView.ResizeToContents)        
    self.table_dragdrop.setAcceptDrops(True)
    self.table_dragdrop.setDragEnabled(True)
    self.table_dragdrop.dropped.connect(self.dataDropped)
    self.table_dragdrop.clicked.connect(self.item_click)
    self.table_dragdrop.itemChanged.connect(self.uncheck_if_zero)
    self.table_dragdrop.doubleClicked.connect(self.item_dclick)
    self.table_dragdrop.itemChanged.connect(self.dataOverviewOn_OnChange)

    self.table_dragdrop.resizeRowsToContents()

    self.table_dragdrop.horizontalHeader().sectionClicked.connect(self.select_all)




    ############################Variables##################################
    #######################################################################
    #Initilaize some variables which are lateron filled in the program
    self.w = None #Initialize a variable for a popup window
    self.threadpool = QtCore.QThreadPool()
    self.threadpool_single = QtCore.QThreadPool()
    self.threadpool_single.setMaxThreadCount(1)
    self.threadpool_single_queue = 0 #count nr. of threads in queue; 

    #self.threadpool_single = QtCore.QThread()
    self.fittingpopups = []  #This app will be designed to allow training of several models ...
    self.fittingpopups_ui = [] #...simultaneously (threading). The info of each model is appended to a list
    self.popupcounter = 0
    self.colorsQt = 10*['red','yellow','blue','cyan','magenta','green','gray','darkRed','darkYellow','darkBlue','darkCyan','darkMagenta','darkGreen','darkGray']    #Some colors which are later used for different subpopulations
    self.model_keras = None #Variable for storing Keras model   
    self.model_keras_path = None
    self.load_model_path = None
    self.loaded_history = None #Variable for storing a loaded history file (for display on History-Tab)
    self.loaded_para = None #Variable for storing a loaded Parameters-file (for display on History-Tab)
    self.plt1 = None #Used for the popup window to display hist and scatter of single experiments
    self.plt2 = None #Used for the history-tab to show accuracy of loaded history files
    self.plt_cm = [] #Used to show images from the interactive Confusion matrix
    self.model_2_convert = None #Variable to store the path to a chosen model (for converting to .nnet)
    self.ram = dict() #Variable to store data if Option "Data to RAM is enabled"
    self.ValidationSet = None
    self.Metrics = dict()
    self.clr_settings = {}
    self.clr_settings["step_size"] = 8 #Number of epochs to fulfill half a cycle
    self.clr_settings["gamma"] = 0.99995 #gamma factor for Exponential decrease method (exp_range)
    self.optimizer_settings = aid_dl.get_optimizer_settings() #the full set of optimizer settings is saved in this variable and might be changed usiung pushButton_optimizer
    
    #self.clip = QtGui.QApplication.clipboard() #This is how one defines a clipboard variable; one can put text on it via:#self.clip.setText("SomeText") 
    self.new_peaks = [] #list to store used defined peaks
    #######################################################################
    #######################################################################

    
    self.gridLayout_8.addWidget(self.table_dragdrop, 0, 0, 1, 1)
    self.splitter_2 = QtWidgets.QSplitter(self.splitter_3)
    self.splitter_2.setOrientation(QtCore.Qt.Horizontal)
    self.splitter_2.setObjectName(_fromUtf8("splitter_2"))
    self.groupBox_DataOverview = QtWidgets.QGroupBox(self.splitter_2)
    self.groupBox_DataOverview.setObjectName(_fromUtf8("groupBox_DataOverview"))
    self.groupBox_DataOverview.setCheckable(True)
    self.groupBox_DataOverview.setChecked(True)
    self.groupBox_DataOverview.toggled.connect(self.dataOverviewOn)
    
    self.gridLayout_5 = QtWidgets.QGridLayout(self.groupBox_DataOverview)
    self.gridLayout_5.setObjectName(_fromUtf8("gridLayout_5"))
    self.tableWidget_Info = QtWidgets.QTableWidget(self.groupBox_DataOverview)
    self.tableWidget_Info.setMinimumSize(QtCore.QSize(0, 0))
    self.tableWidget_Info.setMaximumSize(QtCore.QSize(16777215, 16777215))
    #self.tableWidget_Info.setEditTriggers(QtWidgets.QAbstractItemView.AnyKeyPressed|QtWidgets.QAbstractItemView.DoubleClicked|QtWidgets.QAbstractItemView.EditKeyPressed|QtWidgets.QAbstractItemView.SelectedClicked)
    self.tableWidget_Info.setDragEnabled(False)
    #self.tableWidget_Info.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
    self.tableWidget_Info.setAlternatingRowColors(True)
    self.tableWidget_Info.setObjectName(_fromUtf8("tableWidget_Info"))
    self.tableWidget_Info.setColumnCount(0)
    self.tableWidget_Info.setRowCount(0)
    self.gridLayout_5.addWidget(self.tableWidget_Info, 0, 0, 1, 1)

    self.tabWidget_DefineModel = QtWidgets.QTabWidget(self.splitter_2)
    self.tabWidget_DefineModel.setEnabled(True)
    self.tabWidget_DefineModel.setObjectName("tabWidget_DefineModel")
    self.tab_DefineModel = QtWidgets.QWidget()
    self.tab_DefineModel.setObjectName("tab_DefineModel")
    self.gridLayout_11 = QtWidgets.QGridLayout(self.tab_DefineModel)
    self.gridLayout_11.setObjectName("gridLayout_11")
    self.scrollArea_defineModel = QtWidgets.QScrollArea(self.tab_DefineModel)
    self.scrollArea_defineModel.setWidgetResizable(True)
    self.scrollArea_defineModel.setObjectName("scrollArea_defineModel")
    self.scrollAreaWidgetContents_3 = QtWidgets.QWidget()
    self.scrollAreaWidgetContents_3.setGeometry(QtCore.QRect(0, 0, 598, 192))
    self.scrollAreaWidgetContents_3.setObjectName("scrollAreaWidgetContents_3")
    self.gridLayout_44 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents_3)
    self.gridLayout_44.setObjectName("gridLayout_44")
    self.gridLayout_newLoadModel = QtWidgets.QGridLayout()
    self.gridLayout_newLoadModel.setObjectName("gridLayout_newLoadModel")
    self.verticalLayout_newLoadModel = QtWidgets.QVBoxLayout()
    self.verticalLayout_newLoadModel.setObjectName("verticalLayout_newLoadModel")
    self.radioButton_NewModel = QtWidgets.QRadioButton(self.scrollAreaWidgetContents_3)
    self.radioButton_NewModel.setMinimumSize(QtCore.QSize(0, 20))
    self.radioButton_NewModel.setMaximumSize(QtCore.QSize(16777215, 20))
    self.radioButton_NewModel.setObjectName("radioButton_NewModel")
    self.verticalLayout_newLoadModel.addWidget(self.radioButton_NewModel)
    self.line_loadModel = QtWidgets.QFrame(self.scrollAreaWidgetContents_3)
    self.line_loadModel.setFrameShape(QtWidgets.QFrame.HLine)
    self.line_loadModel.setFrameShadow(QtWidgets.QFrame.Sunken)
    self.line_loadModel.setObjectName("line_loadModel")
    self.verticalLayout_newLoadModel.addWidget(self.line_loadModel)
    self.radioButton_LoadRestartModel = QtWidgets.QRadioButton(self.scrollAreaWidgetContents_3)
    self.radioButton_LoadRestartModel.setMinimumSize(QtCore.QSize(0, 20))
    self.radioButton_LoadRestartModel.setMaximumSize(QtCore.QSize(16777215, 20))
    self.radioButton_LoadRestartModel.setObjectName("radioButton_LoadRestartModel")
    self.radioButton_LoadRestartModel.clicked.connect(self.action_preview_model)
    self.verticalLayout_newLoadModel.addWidget(self.radioButton_LoadRestartModel)
    self.radioButton_LoadContinueModel = QtWidgets.QRadioButton(self.scrollAreaWidgetContents_3)
    self.radioButton_LoadContinueModel.setMinimumSize(QtCore.QSize(0, 20))
    self.radioButton_LoadContinueModel.setMaximumSize(QtCore.QSize(16777215, 20))
    self.radioButton_LoadContinueModel.setObjectName("radioButton_LoadContinueModel")
    self.radioButton_LoadContinueModel.clicked.connect(self.action_preview_model)
    self.verticalLayout_newLoadModel.addWidget(self.radioButton_LoadContinueModel)
    
    self.gridLayout_newLoadModel.addLayout(self.verticalLayout_newLoadModel, 0, 0, 1, 1)
    self.verticalLayout_newLoadModel_2 = QtWidgets.QVBoxLayout()
    self.verticalLayout_newLoadModel_2.setObjectName("verticalLayout_newLoadModel_2")
    self.comboBox_ModelSelection = QtWidgets.QComboBox(self.scrollAreaWidgetContents_3)
    self.comboBox_ModelSelection.setMinimumSize(QtCore.QSize(0, 20))
    self.comboBox_ModelSelection.setMaximumSize(QtCore.QSize(16777215, 20))
    self.comboBox_ModelSelection.setObjectName("comboBox_ModelSelection")
    self.predefined_models = ["None"] + model_zoo.get_predefined_models()
    self.comboBox_ModelSelection.addItems(self.predefined_models)        
    self.verticalLayout_newLoadModel_2.addWidget(self.comboBox_ModelSelection)
    self.line_loadModel_2 = QtWidgets.QFrame(self.scrollAreaWidgetContents_3)
    self.line_loadModel_2.setFrameShape(QtWidgets.QFrame.HLine)
    self.line_loadModel_2.setFrameShadow(QtWidgets.QFrame.Sunken)
    self.line_loadModel_2.setObjectName("line_loadModel_2")
    self.verticalLayout_newLoadModel_2.addWidget(self.line_loadModel_2)
    self.lineEdit_LoadModelPath = QtWidgets.QLineEdit(self.scrollAreaWidgetContents_3)
    self.lineEdit_LoadModelPath.setMinimumSize(QtCore.QSize(0, 40))
    self.lineEdit_LoadModelPath.setMaximumSize(QtCore.QSize(16777215, 40))
    self.lineEdit_LoadModelPath.setObjectName("lineEdit_LoadModelPath")
    self.verticalLayout_newLoadModel_2.addWidget(self.lineEdit_LoadModelPath)
    self.gridLayout_newLoadModel.addLayout(self.verticalLayout_newLoadModel_2, 0, 1, 1, 1)
    self.gridLayout_44.addLayout(self.gridLayout_newLoadModel, 0, 0, 1, 1)
    
    

    
    self.horizontalLayout_modelname = QtWidgets.QHBoxLayout()
    self.horizontalLayout_modelname.setObjectName("horizontalLayout_modelname")
    self.pushButton_modelname = QtWidgets.QPushButton(self.scrollAreaWidgetContents_3)
    self.pushButton_modelname.setObjectName("pushButton_modelname")
    self.pushButton_modelname.clicked.connect(self.action_set_modelpath_and_name)

    self.horizontalLayout_modelname.addWidget(self.pushButton_modelname)
    self.lineEdit_modelname = QtWidgets.QLineEdit(self.scrollAreaWidgetContents_3)
    self.lineEdit_modelname.setMinimumSize(QtCore.QSize(0, 22))
    self.lineEdit_modelname.setMaximumSize(QtCore.QSize(16777215, 22))
    self.lineEdit_modelname.setObjectName("lineEdit_modelname")
    self.horizontalLayout_modelname.addWidget(self.lineEdit_modelname)
    self.gridLayout_44.addLayout(self.horizontalLayout_modelname, 3, 0, 1, 1)
    self.scrollArea_defineModel.setWidget(self.scrollAreaWidgetContents_3)
    self.gridLayout_11.addWidget(self.scrollArea_defineModel, 0, 0, 1, 1)
    self.tabWidget_DefineModel.addTab(self.tab_DefineModel, "")





    self.groupBox_imgProc = QtWidgets.QGroupBox(self.scrollAreaWidgetContents_3)
    self.groupBox_imgProc.setObjectName("groupBox_imgProc")
    self.gridLayout_49 = QtWidgets.QGridLayout(self.groupBox_imgProc)
    self.gridLayout_49.setObjectName("gridLayout_49")
    self.comboBox_Normalization = QtWidgets.QComboBox(self.groupBox_imgProc)
    self.comboBox_Normalization.setMinimumSize(QtCore.QSize(200, 0))
    self.comboBox_Normalization.setObjectName("comboBox_Normalization")
    self.gridLayout_49.addWidget(self.comboBox_Normalization, 0, 4, 1, 1)
    self.spinBox_imagecrop = QtWidgets.QSpinBox(self.groupBox_imgProc)
    self.spinBox_imagecrop.setObjectName("spinBox_imagecrop")
    self.gridLayout_49.addWidget(self.spinBox_imagecrop, 0, 1, 1, 1)
    self.horizontalLayout_crop = QtWidgets.QHBoxLayout()
    self.horizontalLayout_crop.setObjectName("horizontalLayout_crop")
    self.label_CropIcon = QtWidgets.QLabel(self.groupBox_imgProc)
    self.label_CropIcon.setText("")
    #self.label_CropIcon.setPixmap(QtGui.QPixmap("../013_AIDeveloper_0.0.8_dev1/art/Icon theme 1/cropping.png"))
    self.label_CropIcon.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
    self.label_CropIcon.setObjectName("label_CropIcon")
    self.horizontalLayout_crop.addWidget(self.label_CropIcon)
    self.label_Crop = QtWidgets.QLabel(self.groupBox_imgProc)
    self.label_Crop.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
    self.label_Crop.setObjectName("label_Crop")
    self.horizontalLayout_crop.addWidget(self.label_Crop)
    self.gridLayout_49.addLayout(self.horizontalLayout_crop, 0, 0, 1, 1)
    self.comboBox_paddingMode = QtWidgets.QComboBox(self.groupBox_imgProc)
    self.comboBox_paddingMode.setEnabled(True)
    self.comboBox_paddingMode.setObjectName("comboBox_paddingMode")
    self.comboBox_paddingMode.addItem("")
    self.comboBox_paddingMode.addItem("")
    self.comboBox_paddingMode.addItem("")
    self.comboBox_paddingMode.addItem("")
    self.comboBox_paddingMode.addItem("")
    self.comboBox_paddingMode.addItem("")
    self.comboBox_paddingMode.addItem("")
    
    self.gridLayout_49.addWidget(self.comboBox_paddingMode, 1, 1, 1, 1)
    self.horizontalLayout_nrEpochs = QtWidgets.QHBoxLayout()
    self.horizontalLayout_nrEpochs.setObjectName("horizontalLayout_nrEpochs")
    self.label_padIcon = QtWidgets.QLabel(self.groupBox_imgProc)
    self.label_padIcon.setText("")
    #self.label_padIcon.setPixmap(QtGui.QPixmap("../013_AIDeveloper_0.0.8_dev1/art/Icon theme 1/padding.png"))
    self.label_padIcon.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
    self.label_padIcon.setObjectName("label_padIcon")
    self.horizontalLayout_nrEpochs.addWidget(self.label_padIcon)
    self.label_paddingMode = QtWidgets.QLabel(self.groupBox_imgProc)
    self.label_paddingMode.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
    self.label_paddingMode.setObjectName("label_paddingMode")
    self.horizontalLayout_nrEpochs.addWidget(self.label_paddingMode)
    self.gridLayout_49.addLayout(self.horizontalLayout_nrEpochs, 1, 0, 1, 1)
    self.comboBox_GrayOrRGB = QtWidgets.QComboBox(self.groupBox_imgProc)
    self.comboBox_GrayOrRGB.setObjectName("comboBox_GrayOrRGB")
    self.gridLayout_49.addWidget(self.comboBox_GrayOrRGB, 1, 4, 1, 1)
    self.horizontalLayout_colorMode = QtWidgets.QHBoxLayout()
    self.horizontalLayout_colorMode.setObjectName("horizontalLayout_colorMode")
    self.label_colorModeIcon = QtWidgets.QLabel(self.groupBox_imgProc)
    self.label_colorModeIcon.setText("")
    #self.label_colorModeIcon.setPixmap(QtGui.QPixmap("../013_AIDeveloper_0.0.8_dev1/art/Icon theme 1/color_mode.png"))
    self.label_colorModeIcon.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
    self.label_colorModeIcon.setObjectName("label_colorModeIcon")
    self.horizontalLayout_colorMode.addWidget(self.label_colorModeIcon)
    self.label_colorMode = QtWidgets.QLabel(self.groupBox_imgProc)
    self.label_colorMode.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
    self.label_colorMode.setObjectName("label_colorMode")
    self.horizontalLayout_colorMode.addWidget(self.label_colorMode)
    self.gridLayout_49.addLayout(self.horizontalLayout_colorMode, 1, 3, 1, 1)
    self.horizontalLayout_normalization = QtWidgets.QHBoxLayout()
    self.horizontalLayout_normalization.setObjectName("horizontalLayout_normalization")
    self.label_NormalizationIcon = QtWidgets.QLabel(self.groupBox_imgProc)
    self.label_NormalizationIcon.setText("")
    #self.label_NormalizationIcon.setPixmap(QtGui.QPixmap("../013_AIDeveloper_0.0.8_dev1/art/Icon theme 1/normalization.png"))
    self.label_NormalizationIcon.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
    self.label_NormalizationIcon.setObjectName("label_NormalizationIcon")
    self.horizontalLayout_normalization.addWidget(self.label_NormalizationIcon)
    self.label_Normalization = QtWidgets.QLabel(self.groupBox_imgProc)
    self.label_Normalization.setLayoutDirection(QtCore.Qt.LeftToRight)
    self.label_Normalization.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
    self.label_Normalization.setObjectName("label_Normalization")
    self.horizontalLayout_normalization.addWidget(self.label_Normalization)
    self.gridLayout_49.addLayout(self.horizontalLayout_normalization, 0, 3, 1, 1)
    self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
    self.horizontalLayout_2.setObjectName("horizontalLayout_2")
    self.label_zoomIcon = QtWidgets.QLabel(self.groupBox_imgProc)
    self.label_zoomIcon.setText("")
    #self.label_zoomIcon.setPixmap(QtGui.QPixmap("../000_Icons/Version_2/zoom_order.png"))
    self.label_zoomIcon.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
    self.label_zoomIcon.setObjectName("label_zoomIcon")
    self.horizontalLayout_2.addWidget(self.label_zoomIcon)
    self.label_zoomOrder = QtWidgets.QLabel(self.groupBox_imgProc)
    self.label_zoomOrder.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
    self.label_zoomOrder.setObjectName("label_zoomOrder")
    self.horizontalLayout_2.addWidget(self.label_zoomOrder)
    self.gridLayout_49.addLayout(self.horizontalLayout_2, 2, 0, 1, 1)
    self.comboBox_zoomOrder = QtWidgets.QComboBox(self.groupBox_imgProc)
    self.comboBox_zoomOrder.setObjectName("comboBox_zoomOrder")
    self.comboBox_zoomOrder.addItem("")
    self.comboBox_zoomOrder.addItem("")
    self.comboBox_zoomOrder.addItem("")
    self.comboBox_zoomOrder.addItem("")
    self.comboBox_zoomOrder.addItem("")
    self.comboBox_zoomOrder.setMaximumSize(QtCore.QSize(100, 16777215))
    
    self.gridLayout_49.addWidget(self.comboBox_zoomOrder, 2, 1, 1, 1)
    self.gridLayout_44.addWidget(self.groupBox_imgProc, 1, 0, 1, 1)





    self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
    self.horizontalLayout_4.setObjectName("horizontalLayout_4")
    self.groupBox_system = QtWidgets.QGroupBox(self.scrollAreaWidgetContents_3)
    self.groupBox_system.setObjectName("groupBox_system")
    self.gridLayout_48 = QtWidgets.QGridLayout(self.groupBox_system)
    self.gridLayout_48.setObjectName("gridLayout_48")
    self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
    self.horizontalLayout_5.setObjectName("horizontalLayout_5")
    self.label_nrEpochsIcon = QtWidgets.QLabel(self.groupBox_system)
    self.label_nrEpochsIcon.setText("")
    self.label_nrEpochsIcon.setObjectName("label_nrEpochsIcon")
    self.horizontalLayout_5.addWidget(self.label_nrEpochsIcon)
    self.label_nrEpochs = QtWidgets.QLabel(self.groupBox_system)
    self.label_nrEpochs.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
    self.label_nrEpochs.setObjectName("label_nrEpochs")
    self.horizontalLayout_5.addWidget(self.label_nrEpochs)
    self.spinBox_NrEpochs = QtWidgets.QSpinBox(self.groupBox_system)
    self.spinBox_NrEpochs.setObjectName("spinBox_NrEpochs")
    self.horizontalLayout_5.addWidget(self.spinBox_NrEpochs)
    self.gridLayout_48.addLayout(self.horizontalLayout_5, 0, 0, 1, 1)
    self.line_nrEpochs_cpu = QtWidgets.QFrame(self.groupBox_system)
    self.line_nrEpochs_cpu.setFrameShape(QtWidgets.QFrame.VLine)
    self.line_nrEpochs_cpu.setFrameShadow(QtWidgets.QFrame.Sunken)
    self.line_nrEpochs_cpu.setObjectName("line_nrEpochs_cpu")
    self.gridLayout_48.addWidget(self.line_nrEpochs_cpu, 0, 1, 2, 1)
    self.radioButton_cpu = QtWidgets.QRadioButton(self.groupBox_system)
    self.radioButton_cpu.setObjectName("radioButton_cpu")
    self.gridLayout_48.addWidget(self.radioButton_cpu, 0, 2, 1, 1)
    self.comboBox_cpu = QtWidgets.QComboBox(self.groupBox_system)
    self.comboBox_cpu.setObjectName("comboBox_cpu")
    self.gridLayout_48.addWidget(self.comboBox_cpu, 0, 3, 1, 1)
    spacerItem = QtWidgets.QSpacerItem(198, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
    self.gridLayout_48.addItem(spacerItem, 0, 4, 1, 3)
    spacerItem1 = QtWidgets.QSpacerItem(211, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
    self.gridLayout_48.addItem(spacerItem1, 1, 0, 1, 1)
    self.radioButton_gpu = QtWidgets.QRadioButton(self.groupBox_system)
    self.radioButton_gpu.setObjectName("radioButton_gpu")
    self.gridLayout_48.addWidget(self.radioButton_gpu, 1, 2, 1, 1)
    self.comboBox_gpu = QtWidgets.QComboBox(self.groupBox_system)
    self.comboBox_gpu.setObjectName("comboBox_gpu")
    self.gridLayout_48.addWidget(self.comboBox_gpu, 1, 3, 1, 2)
    self.label_memory = QtWidgets.QLabel(self.groupBox_system)
    self.label_memory.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
    self.label_memory.setObjectName("label_memory")
    self.gridLayout_48.addWidget(self.label_memory, 1, 5, 1, 1)
    self.doubleSpinBox_memory = QtWidgets.QDoubleSpinBox(self.groupBox_system)
    self.doubleSpinBox_memory.setObjectName("doubleSpinBox_memory")
    self.gridLayout_48.addWidget(self.doubleSpinBox_memory, 1, 6, 1, 1)
    self.horizontalLayout_4.addWidget(self.groupBox_system)
    
    
    
    
    
    
    
    
    self.gridLayout_44.addLayout(self.horizontalLayout_4, 2, 0, 1, 1)
    #############Manual settings##############
    #self.label_colorMode.setMinimumSize(QtCore.QSize(55,22))
    self.label_nrEpochsIcon.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
    self.label_nrEpochs.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
    #self.label_nrEpochs.setMaximumSize(QtCore.QSize(50, 22))

    #self.label_NormalizationIcon.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
    #self.label_NormalizationIcon.setText("")
    #self.label_Normalization.setLayoutDirection(QtCore.Qt.LeftToRight)
    #self.label_Normalization.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
    #self.label_CropSpace.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
    #self.label_CropIcon.setMinimumSize(QtCore.QSize(20, 20))
    #self.label_CropIcon.setMaximumSize(QtCore.QSize(20, 20))

    self.comboBox_Normalization.setMinimumSize(QtCore.QSize(200,22))
    self.comboBox_Normalization.setMaximumSize(QtCore.QSize(200, 22))
    self.norm_methods = Default_dict["norm_methods"]
    self.comboBox_Normalization.addItems(self.norm_methods)
    width=self.comboBox_Normalization.fontMetrics().boundingRect(max(self.norm_methods, key=len)).width()
    self.comboBox_Normalization.view().setFixedWidth(width+10)             
    self.spinBox_imagecrop.setMinimum(1)
    self.spinBox_imagecrop.setMaximum(9E8)
    self.spinBox_NrEpochs.setMinimum(1)
    self.spinBox_NrEpochs.setMaximum(9E8)

    self.comboBox_gpu.setEnabled(False)
    self.doubleSpinBox_memory.setEnabled(False)
    self.doubleSpinBox_memory.setValue(0.7)
    self.comboBox_ModelSelection.setEnabled(False)
    self.lineEdit_LoadModelPath.setEnabled(False)

    self.radioButton_gpu.toggled['bool'].connect(self.comboBox_gpu.setEnabled)
    self.radioButton_gpu.toggled['bool'].connect(self.label_memory.setEnabled)
    self.radioButton_gpu.toggled['bool'].connect(self.doubleSpinBox_memory.setEnabled)
    self.radioButton_cpu.toggled['bool'].connect(self.comboBox_cpu.setEnabled)
    self.radioButton_NewModel.toggled['bool'].connect(self.comboBox_ModelSelection.setEnabled)
    self.radioButton_LoadRestartModel.toggled['bool'].connect(self.lineEdit_LoadModelPath.setEnabled)
    self.radioButton_LoadContinueModel.toggled['bool'].connect(self.lineEdit_LoadModelPath.setEnabled)

    
    self.tab_kerasAug = QtWidgets.QWidget()
    self.tab_kerasAug.setObjectName(_fromUtf8("tab_kerasAug"))
    self.gridLayout_7 = QtWidgets.QGridLayout(self.tab_kerasAug)
    self.gridLayout_7.setObjectName(_fromUtf8("gridLayout_7"))
    self.verticalLayout_8 = QtWidgets.QVBoxLayout()
    self.verticalLayout_8.setObjectName(_fromUtf8("verticalLayout_8"))
    self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
    self.horizontalLayout_9.setObjectName(_fromUtf8("horizontalLayout_9"))
    self.label_RefreshAfterEpochs = QtWidgets.QLabel(self.tab_kerasAug)
    self.label_RefreshAfterEpochs.setObjectName(_fromUtf8("label_RefreshAfterEpochs"))
    self.horizontalLayout_9.addWidget(self.label_RefreshAfterEpochs)
    self.spinBox_RefreshAfterEpochs = QtWidgets.QSpinBox(self.tab_kerasAug)
    self.spinBox_RefreshAfterEpochs.setObjectName(_fromUtf8("spinBox_RefreshAfterEpochs"))
    self.spinBox_RefreshAfterEpochs.setMinimum(1)
    self.spinBox_RefreshAfterEpochs.setMaximum(9E8)
    self.horizontalLayout_9.addWidget(self.spinBox_RefreshAfterEpochs)
    self.verticalLayout_8.addLayout(self.horizontalLayout_9)
    self.verticalLayout_7 = QtWidgets.QVBoxLayout()
    self.verticalLayout_7.setObjectName(_fromUtf8("verticalLayout_7"))
    self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
    self.horizontalLayout_8.setObjectName(_fromUtf8("horizontalLayout_8"))
    self.checkBox_HorizFlip = QtWidgets.QCheckBox(self.tab_kerasAug)
    self.checkBox_HorizFlip.setObjectName(_fromUtf8("checkBox_HorizFlip"))
    self.horizontalLayout_8.addWidget(self.checkBox_HorizFlip)
    self.checkBox_VertFlip = QtWidgets.QCheckBox(self.tab_kerasAug)
    self.checkBox_VertFlip.setObjectName(_fromUtf8("checkBox_VertFlip"))
    self.horizontalLayout_8.addWidget(self.checkBox_VertFlip)
    self.verticalLayout_7.addLayout(self.horizontalLayout_8)
    self.splitter = QtWidgets.QSplitter(self.tab_kerasAug)
    self.splitter.setOrientation(QtCore.Qt.Horizontal)
    self.splitter.setObjectName(_fromUtf8("splitter"))
    self.widget = QtWidgets.QWidget(self.splitter)
    self.widget.setObjectName(_fromUtf8("widget"))
    self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.widget)
    self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
    
    self.onlyFloat = QtGui.QDoubleValidator()
    
    self.verticalLayout_6.setObjectName(_fromUtf8("verticalLayout_6"))
    self.label_Rotation = QtWidgets.QCheckBox(self.widget)
    self.label_Rotation.setObjectName(_fromUtf8("label_Rotation"))
    self.verticalLayout_6.addWidget(self.label_Rotation)
    self.label_width_shift = QtWidgets.QCheckBox(self.widget)
    self.label_width_shift.setObjectName(_fromUtf8("label_width_shift"))
    self.verticalLayout_6.addWidget(self.label_width_shift)
    self.label_height_shift = QtWidgets.QCheckBox(self.widget)
    self.label_height_shift.setObjectName(_fromUtf8("label_height_shift"))
    self.verticalLayout_6.addWidget(self.label_height_shift)
    self.label_zoom = QtWidgets.QCheckBox(self.widget)
    self.label_zoom.setObjectName(_fromUtf8("label_zoom"))
    self.verticalLayout_6.addWidget(self.label_zoom)
    self.label_shear = QtWidgets.QCheckBox(self.widget)
    self.label_shear.setObjectName(_fromUtf8("label_shear"))
    self.verticalLayout_6.addWidget(self.label_shear)
    self.widget1 = QtWidgets.QWidget(self.splitter)
    self.widget1.setObjectName(_fromUtf8("widget1"))
    self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.widget1)
    self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
    self.verticalLayout_5.setObjectName(_fromUtf8("verticalLayout_5"))
    self.lineEdit_Rotation = QtWidgets.QLineEdit(self.widget1)
    self.lineEdit_Rotation.setObjectName(_fromUtf8("lineEdit_Rotation"))
    self.lineEdit_Rotation.setValidator(self.onlyFloat)
    self.verticalLayout_5.addWidget(self.lineEdit_Rotation)
    self.lineEdit_widthShift = QtWidgets.QLineEdit(self.widget1)
    self.lineEdit_widthShift.setObjectName(_fromUtf8("lineEdit_widthShift"))
    self.lineEdit_widthShift.setValidator(self.onlyFloat)

    self.verticalLayout_5.addWidget(self.lineEdit_widthShift)
    self.lineEdit_heightShift = QtWidgets.QLineEdit(self.widget1)
    self.lineEdit_heightShift.setObjectName(_fromUtf8("lineEdit_heightShift"))
    self.lineEdit_heightShift.setValidator(self.onlyFloat)

    self.verticalLayout_5.addWidget(self.lineEdit_heightShift)
    self.lineEdit_zoomRange = QtWidgets.QLineEdit(self.widget1)
    self.lineEdit_zoomRange.setObjectName(_fromUtf8("lineEdit_zoomRange"))
    self.lineEdit_zoomRange.setValidator(self.onlyFloat)

    self.verticalLayout_5.addWidget(self.lineEdit_zoomRange)
    self.lineEdit_shearRange = QtWidgets.QLineEdit(self.widget1)
    self.lineEdit_shearRange.setObjectName(_fromUtf8("lineEdit_shearRange"))
    self.lineEdit_shearRange.setValidator(self.onlyFloat)

    self.verticalLayout_5.addWidget(self.lineEdit_shearRange)
    self.verticalLayout_7.addWidget(self.splitter)
    self.verticalLayout_8.addLayout(self.verticalLayout_7)
    self.gridLayout_7.addLayout(self.verticalLayout_8, 0, 0, 1, 1)
    self.tabWidget_DefineModel.addTab(self.tab_kerasAug, _fromUtf8(""))
    
    
    
    
    
    self.tab_BrightnessAug = QtWidgets.QWidget()
    self.tab_BrightnessAug.setObjectName("tab_BrightnessAug")
    self.gridLayout_42 = QtWidgets.QGridLayout(self.tab_BrightnessAug)
    self.gridLayout_42.setObjectName("gridLayout_42")
    self.scrollArea_2 = QtWidgets.QScrollArea(self.tab_BrightnessAug)
    self.scrollArea_2.setWidgetResizable(True)
    self.scrollArea_2.setObjectName("scrollArea_2")
    self.scrollAreaWidgetContents_2 = QtWidgets.QWidget()
    self.scrollAreaWidgetContents_2.setGeometry(QtCore.QRect(0, -2, 449, 269))
    self.scrollAreaWidgetContents_2.setObjectName("scrollAreaWidgetContents_2")
    self.gridLayout_43 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents_2)
    self.gridLayout_43.setObjectName("gridLayout_43")
    self.groupBox_GaussianNoise = QtWidgets.QGroupBox(self.scrollAreaWidgetContents_2)
    self.groupBox_GaussianNoise.setObjectName("groupBox_GaussianNoise")
    self.gridLayout_13 = QtWidgets.QGridLayout(self.groupBox_GaussianNoise)
    self.gridLayout_13.setObjectName("gridLayout_13")
    self.label_GaussianNoiseMean = QtWidgets.QCheckBox(self.groupBox_GaussianNoise)
    self.label_GaussianNoiseMean.setObjectName("label_GaussianNoiseMean")
    self.gridLayout_13.addWidget(self.label_GaussianNoiseMean, 0, 0, 1, 1)
    self.doubleSpinBox_GaussianNoiseMean = QtWidgets.QDoubleSpinBox(self.groupBox_GaussianNoise)
    self.doubleSpinBox_GaussianNoiseMean.setObjectName("spinBox_GaussianNoiseMean")
    self.gridLayout_13.addWidget(self.doubleSpinBox_GaussianNoiseMean, 0, 1, 1, 1)
    self.label_GaussianNoiseScale = QtWidgets.QCheckBox(self.groupBox_GaussianNoise)
    self.label_GaussianNoiseScale.setObjectName("label_GaussianNoiseScale")
    self.gridLayout_13.addWidget(self.label_GaussianNoiseScale, 1, 0, 1, 1)
    self.doubleSpinBox_GaussianNoiseScale = QtWidgets.QDoubleSpinBox(self.groupBox_GaussianNoise)
    self.doubleSpinBox_GaussianNoiseScale.setObjectName("spinBox_GaussianNoiseScale")
    self.gridLayout_13.addWidget(self.doubleSpinBox_GaussianNoiseScale, 1, 1, 1, 1)
    self.gridLayout_43.addWidget(self.groupBox_GaussianNoise, 2, 1, 1, 1)
    self.groupBox_colorAugmentation = QtWidgets.QGroupBox(self.scrollAreaWidgetContents_2)
    self.groupBox_colorAugmentation.setCheckable(False)
    self.groupBox_colorAugmentation.setObjectName("groupBox_colorAugmentation")
    self.gridLayout_15 = QtWidgets.QGridLayout(self.groupBox_colorAugmentation)
    self.gridLayout_15.setObjectName("gridLayout_15")
    self.doubleSpinBox_contrastLower = QtWidgets.QDoubleSpinBox(self.groupBox_colorAugmentation)
    self.doubleSpinBox_contrastLower.setObjectName("doubleSpinBox_contrastLower")
    self.doubleSpinBox_contrastLower.setMaximumSize(QtCore.QSize(75, 16777215))

    self.gridLayout_15.addWidget(self.doubleSpinBox_contrastLower, 0, 1, 1, 1)
    self.doubleSpinBox_saturationHigher = QtWidgets.QDoubleSpinBox(self.groupBox_colorAugmentation)
    self.doubleSpinBox_saturationHigher.setObjectName("doubleSpinBox_saturationHigher")
    self.doubleSpinBox_saturationHigher.setMaximumSize(QtCore.QSize(75, 16777215))

    self.gridLayout_15.addWidget(self.doubleSpinBox_saturationHigher, 1, 2, 1, 1)
    self.doubleSpinBox_contrastHigher = QtWidgets.QDoubleSpinBox(self.groupBox_colorAugmentation)
    self.doubleSpinBox_contrastHigher.setObjectName("doubleSpinBox_contrastHigher")
    self.doubleSpinBox_contrastHigher.setMaximumSize(QtCore.QSize(75, 16777215))

    self.gridLayout_15.addWidget(self.doubleSpinBox_contrastHigher, 0, 2, 1, 1)
    self.checkBox_contrast = QtWidgets.QCheckBox(self.groupBox_colorAugmentation)
    self.checkBox_contrast.setCheckable(True)
    self.checkBox_contrast.setObjectName("checkBox_contrast")
    self.gridLayout_15.addWidget(self.checkBox_contrast, 0, 0, 1, 1)
    self.doubleSpinBox_saturationLower = QtWidgets.QDoubleSpinBox(self.groupBox_colorAugmentation)
    self.doubleSpinBox_saturationLower.setObjectName("doubleSpinBox_saturationLower")
    self.doubleSpinBox_saturationLower.setMaximumSize(QtCore.QSize(75, 16777215))

    self.gridLayout_15.addWidget(self.doubleSpinBox_saturationLower, 1, 1, 1, 1)
    self.doubleSpinBox_hueDelta = QtWidgets.QDoubleSpinBox(self.groupBox_colorAugmentation)
    self.doubleSpinBox_hueDelta.setObjectName("doubleSpinBox_hueDelta")
    self.doubleSpinBox_hueDelta.setMaximumSize(QtCore.QSize(75, 16777215))
    
    self.gridLayout_15.addWidget(self.doubleSpinBox_hueDelta, 2, 1, 1, 1)
    self.checkBox_saturation = QtWidgets.QCheckBox(self.groupBox_colorAugmentation)
    self.checkBox_saturation.setObjectName("checkBox_saturation")
    self.gridLayout_15.addWidget(self.checkBox_saturation, 1, 0, 1, 1)
    self.checkBox_hue = QtWidgets.QCheckBox(self.groupBox_colorAugmentation)
    self.checkBox_hue.setObjectName("checkBox_hue")
    self.gridLayout_15.addWidget(self.checkBox_hue, 2, 0, 1, 1)
    self.gridLayout_43.addWidget(self.groupBox_colorAugmentation, 3, 0, 1, 1)
    
    self.groupBox_blurringAug = QtWidgets.QGroupBox(self.scrollAreaWidgetContents_2)
    self.groupBox_blurringAug.setObjectName("groupBox_blurringAug")
    self.gridLayout_45 = QtWidgets.QGridLayout(self.groupBox_blurringAug)
    self.gridLayout_45.setObjectName("gridLayout_45")
    self.gridLayout_blur = QtWidgets.QGridLayout()
    self.gridLayout_blur.setObjectName("gridLayout_blur")
    self.label_avgBlurMin = QtWidgets.QLabel(self.groupBox_blurringAug)
    self.label_avgBlurMin.setMaximumSize(QtCore.QSize(31, 16777215))
    self.label_avgBlurMin.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
    self.label_avgBlurMin.setObjectName("label_avgBlurMin")
    self.gridLayout_blur.addWidget(self.label_avgBlurMin, 0, 1, 1, 1)
    self.spinBox_avgBlurMin = QtWidgets.QSpinBox(self.groupBox_blurringAug)
    self.spinBox_avgBlurMin.setObjectName("spinBox_avgBlurMin")
    self.gridLayout_blur.addWidget(self.spinBox_avgBlurMin, 0, 2, 1, 1)
    self.spinBox_avgBlurMax = QtWidgets.QSpinBox(self.groupBox_blurringAug)
    self.spinBox_avgBlurMax.setObjectName("spinBox_avgBlurMax")
    self.gridLayout_blur.addWidget(self.spinBox_avgBlurMax, 0, 4, 1, 1)
    self.checkBox_avgBlur = QtWidgets.QCheckBox(self.groupBox_blurringAug)
    self.checkBox_avgBlur.setObjectName("checkBox_avgBlur")
    self.gridLayout_blur.addWidget(self.checkBox_avgBlur, 0, 0, 1, 1)
    self.label_avgBlurMax = QtWidgets.QLabel(self.groupBox_blurringAug)
    self.label_avgBlurMax.setMaximumSize(QtCore.QSize(31, 16777215))
    self.label_avgBlurMax.setObjectName("label_avgBlurMax")
    self.gridLayout_blur.addWidget(self.label_avgBlurMax, 0, 3, 1, 1)
    self.checkBox_avgBlur.setCheckable(True)
    self.spinBox_gaussBlurMax = QtWidgets.QSpinBox(self.groupBox_blurringAug)
    self.spinBox_gaussBlurMax.setObjectName("spinBox_gaussBlurMax")
    self.gridLayout_blur.addWidget(self.spinBox_gaussBlurMax, 1, 4, 1, 1)
    self.spinBox_gaussBlurMin = QtWidgets.QSpinBox(self.groupBox_blurringAug)
    self.spinBox_gaussBlurMin.setObjectName("spinBox_gaussBlurMin")
    self.gridLayout_blur.addWidget(self.spinBox_gaussBlurMin, 1, 2, 1, 1)
    self.label_gaussBlurMin = QtWidgets.QLabel(self.groupBox_blurringAug)
    self.label_gaussBlurMin.setMaximumSize(QtCore.QSize(31, 16777215))
    self.label_gaussBlurMin.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
    self.label_gaussBlurMin.setObjectName("label_gaussBlurMin")
    self.gridLayout_blur.addWidget(self.label_gaussBlurMin, 1, 1, 1, 1)
    self.checkBox_gaussBlur = QtWidgets.QCheckBox(self.groupBox_blurringAug)
    self.checkBox_gaussBlur.setObjectName("checkBox_gaussBlur")
    self.gridLayout_blur.addWidget(self.checkBox_gaussBlur, 1, 0, 1, 1)
    self.label_gaussBlurMax = QtWidgets.QLabel(self.groupBox_blurringAug)
    self.label_gaussBlurMax.setMaximumSize(QtCore.QSize(31, 16777215))
    self.label_gaussBlurMax.setObjectName("label_gaussBlurMax")
    self.gridLayout_blur.addWidget(self.label_gaussBlurMax, 1, 3, 1, 1)
    self.checkBox_gaussBlur.setCheckable(True)
    self.label_motionBlurKernel = QtWidgets.QLabel(self.groupBox_blurringAug)
    self.label_motionBlurKernel.setMaximumSize(QtCore.QSize(31, 16777215))
    self.label_motionBlurKernel.setObjectName("label_motionBlurKernel")
    self.gridLayout_blur.addWidget(self.label_motionBlurKernel, 2, 1, 1, 1)
    self.lineEdit_motionBlurAngle = QtWidgets.QLineEdit(self.groupBox_blurringAug)
    validator = QtGui.QRegExpValidator(QtCore.QRegExp("^[-+]?[0-9]\\d{0,3},(\\d{3})$"))
    self.lineEdit_motionBlurAngle.setValidator(validator)
    self.lineEdit_motionBlurAngle.setMaximumSize(QtCore.QSize(100, 16777215))
    self.lineEdit_motionBlurAngle.setInputMask("")
    self.lineEdit_motionBlurAngle.setObjectName("lineEdit_motionBlurAngle")
    self.gridLayout_blur.addWidget(self.lineEdit_motionBlurAngle, 2, 4, 1, 1)
    self.checkBox_motionBlur = QtWidgets.QCheckBox(self.groupBox_blurringAug)
    self.checkBox_motionBlur.setMaximumSize(QtCore.QSize(100, 16777215))
    self.checkBox_motionBlur.setObjectName("checkBox_motionBlur")
    self.gridLayout_blur.addWidget(self.checkBox_motionBlur, 2, 0, 1, 1)
    self.label_motionBlurAngle = QtWidgets.QLabel(self.groupBox_blurringAug)
    self.label_motionBlurAngle.setMaximumSize(QtCore.QSize(16777215, 16777215))
    self.label_motionBlurAngle.setObjectName("label_motionBlurAngle")
    self.gridLayout_blur.addWidget(self.label_motionBlurAngle, 2, 3, 1, 1)
    validator = QtGui.QRegExpValidator(QtCore.QRegExp("^\\d{1,3},(\\d{3})$"))
    self.lineEdit_motionBlurKernel = QtWidgets.QLineEdit(self.groupBox_blurringAug)
    self.lineEdit_motionBlurKernel.setValidator(validator)
    self.lineEdit_motionBlurKernel.setMaximumSize(QtCore.QSize(100, 16777215))
    self.lineEdit_motionBlurKernel.setInputMask("")
    self.lineEdit_motionBlurKernel.setMaxLength(32767)
    self.lineEdit_motionBlurKernel.setObjectName("lineEdit_motionBlurKernel")
    self.gridLayout_blur.addWidget(self.lineEdit_motionBlurKernel, 2, 2, 1, 1)
    self.gridLayout_45.addLayout(self.gridLayout_blur, 0, 0, 1, 1)
    self.gridLayout_43.addWidget(self.groupBox_blurringAug, 3, 1, 1, 1)
    self.checkBox_motionBlur.setCheckable(True)
    self.lineEdit_motionBlurKernel.setClearButtonEnabled(False)        
    


 
    self.groupBox_BrightnessAugmentation = QtWidgets.QGroupBox(self.scrollAreaWidgetContents_2)
    self.groupBox_BrightnessAugmentation.setObjectName("groupBox_BrightnessAugmentation")
    self.gridLayout_12 = QtWidgets.QGridLayout(self.groupBox_BrightnessAugmentation)
    self.gridLayout_12.setObjectName("gridLayout_12")
    self.label_Plus = QtWidgets.QCheckBox(self.groupBox_BrightnessAugmentation)
    self.label_Plus.setObjectName("label_Plus")
    self.gridLayout_12.addWidget(self.label_Plus, 1, 0, 1, 1)
    self.doubleSpinBox_MultLower = QtWidgets.QDoubleSpinBox(self.groupBox_BrightnessAugmentation)
    self.doubleSpinBox_MultLower.setMaximumSize(QtCore.QSize(75, 16777215))
    self.doubleSpinBox_MultLower.setObjectName("doubleSpinBox_MultLower")
    self.gridLayout_12.addWidget(self.doubleSpinBox_MultLower, 2, 1, 1, 1)
    self.spinBox_PlusUpper = QtWidgets.QSpinBox(self.groupBox_BrightnessAugmentation)
    self.spinBox_PlusUpper.setObjectName("spinBox_PlusUpper")
    self.spinBox_PlusUpper.setMaximumSize(QtCore.QSize(75, 16777215))

    self.gridLayout_12.addWidget(self.spinBox_PlusUpper, 1, 2, 1, 2)
    self.spinBox_PlusLower = QtWidgets.QSpinBox(self.groupBox_BrightnessAugmentation)
    self.spinBox_PlusLower.setObjectName("spinBox_PlusLower")
    self.spinBox_PlusLower.setMaximumSize(QtCore.QSize(75, 16777215))

    self.gridLayout_12.addWidget(self.spinBox_PlusLower, 1, 1, 1, 1)
    self.label_Mult = QtWidgets.QCheckBox(self.groupBox_BrightnessAugmentation)
    self.label_Mult.setObjectName("label_Mult")
    self.gridLayout_12.addWidget(self.label_Mult, 2, 0, 1, 1)
    self.doubleSpinBox_MultUpper = QtWidgets.QDoubleSpinBox(self.groupBox_BrightnessAugmentation)
    self.doubleSpinBox_MultUpper.setObjectName("doubleSpinBox_MultUpper")
    self.doubleSpinBox_MultUpper.setMaximumSize(QtCore.QSize(75, 16777215))
    self.gridLayout_12.addWidget(self.doubleSpinBox_MultUpper, 2, 2, 1, 2)
    self.gridLayout_43.addWidget(self.groupBox_BrightnessAugmentation, 2, 0, 1, 1)
    self.spinBox_RefreshAfterNrEpochs = QtWidgets.QSpinBox(self.scrollAreaWidgetContents_2)
    self.spinBox_RefreshAfterNrEpochs.setObjectName("spinBox_RefreshAfterNrEpochs")
    self.gridLayout_43.addWidget(self.spinBox_RefreshAfterNrEpochs, 1, 1, 1, 1)
    self.label_RefreshAfterNrEpochs = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
    self.label_RefreshAfterNrEpochs.setObjectName("label_RefreshAfterNrEpochs")
    self.gridLayout_43.addWidget(self.label_RefreshAfterNrEpochs, 1, 0, 1, 1)
    self.scrollArea_2.setWidget(self.scrollAreaWidgetContents_2)
    self.gridLayout_42.addWidget(self.scrollArea_2, 0, 0, 1, 1)
    self.tabWidget_DefineModel.addTab(self.tab_BrightnessAug, "")
    

    #################################ICONS#################################
    #use full ABSOLUTE path to the image, not relative
    self.radioButton_NewModel.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"model_new.png")))
    self.radioButton_LoadRestartModel.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"model_restart.png")))
    self.radioButton_LoadContinueModel.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"model_continue.png")))
    self.label_CropIcon.setPixmap(QtGui.QPixmap(os.path.join(dir_root,"art",Default_dict["Icon theme"],"cropping.png")))
    self.label_CropIcon.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
    self.pushButton_modelname.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"model_path.png")))

    icon = QtGui.QIcon()
    icon.addPixmap(QtGui.QPixmap(os.path.join(dir_root,"art",Default_dict["Icon theme"],"cpu.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
    self.radioButton_cpu.setIcon(icon)
    self.radioButton_cpu.setEnabled(True)
    icon = QtGui.QIcon()
    icon.addPixmap(QtGui.QPixmap(os.path.join(dir_root,"art",Default_dict["Icon theme"],"gpu.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
    self.radioButton_gpu.setIcon(icon)

    self.label_colorModeIcon.setPixmap(QtGui.QPixmap(os.path.join(dir_root,"art",Default_dict["Icon theme"],"color_mode.png")))
    self.label_NormalizationIcon.setPixmap(QtGui.QPixmap(os.path.join(dir_root,"art",Default_dict["Icon theme"],"normalization.png")))
    self.label_padIcon.setPixmap(QtGui.QPixmap(os.path.join(dir_root,"art",Default_dict["Icon theme"],"padding.png")))
    self.label_zoomIcon.setPixmap(QtGui.QPixmap(os.path.join(dir_root,"art",Default_dict["Icon theme"],"zoom_order.png")))
    
    self.label_nrEpochsIcon.setPixmap(QtGui.QPixmap(os.path.join(dir_root,"art",Default_dict["Icon theme"],"nr_epochs.png")))

    self.checkBox_HorizFlip.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"horizontal_flip.png")))
    self.checkBox_VertFlip.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"vertical_flip.png")))
    self.label_Rotation.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"rotation.png")))
    self.label_Rotation.setChecked(True)
    self.label_Rotation.stateChanged.connect(self.keras_changed_rotation)
    self.label_width_shift.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"width_shift.png")))
    self.label_width_shift.setChecked(True)
    self.label_width_shift.stateChanged.connect(self.keras_changed_width_shift)
    self.label_height_shift.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"height_shift.png")))
    self.label_height_shift.setChecked(True)
    self.label_height_shift.stateChanged.connect(self.keras_changed_height_shift)
    self.label_zoom.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"zoom.png")))
    self.label_zoom.setChecked(True)
    self.label_zoom.stateChanged.connect(self.keras_changed_zoom)
    self.label_shear.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"shear.png")))
    self.label_shear.setChecked(True)
    self.label_shear.stateChanged.connect(self.keras_changed_shear)
    self.label_Plus.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"brightness_plus.png")))
    self.label_Plus.setChecked(True)
    self.label_Plus.stateChanged.connect(self.keras_changed_brightplus)
    self.label_Mult.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"brightness_mult.png")))
    self.label_Mult.setChecked(True)
    self.label_Mult.stateChanged.connect(self.keras_changed_brightmult)
    self.label_GaussianNoiseMean.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"gaussian_noise_mean.png")))
    self.label_GaussianNoiseMean.setChecked(True)
    self.label_GaussianNoiseMean.stateChanged.connect(self.keras_changed_noiseMean)
    self.label_GaussianNoiseScale.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"gaussian_noise_scale.png")))
    self.label_GaussianNoiseScale.setChecked(True)
    self.label_GaussianNoiseScale.stateChanged.connect(self.keras_changed_noiseScale)
    self.checkBox_contrast.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"contrast.png")))
    self.checkBox_contrast.stateChanged.connect(self.keras_changed_contrast)
    self.checkBox_saturation.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"saturation.png")))
    self.checkBox_saturation.stateChanged.connect(self.keras_changed_saturation)
    self.checkBox_hue.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"hue.png")))
    self.checkBox_hue.stateChanged.connect(self.keras_changed_hue)
    self.checkBox_avgBlur.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"average_blur.png")))
    #self.checkBox_avgBlur.stateChanged.connect(self.changed_averageBlur)
    self.checkBox_gaussBlur.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"gaussian_blur.png")))
    #self.checkBox_gaussBlur.stateChanged.connect(self.changed_gaussBlur)
    self.checkBox_motionBlur.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"motion_blur.png")))
    #self.checkBox_motionBlur.stateChanged.connect(self.changed_motionBlur)




    #There will be text on the label_colorMode (Color Mode), find out how "long" the text is and 
    #resize the label to that
    width=self.label_colorMode.fontMetrics().boundingRect(max(["Color Mode"], key=len)).width()
    height = self.label_colorMode.geometry().height()
    self.label_colorMode.setMaximumSize(QtCore.QSize(width, height))


    #Manual values on Build Model Tab
    self.spinBox_PlusUpper.setMinimum(-255)
    self.spinBox_PlusUpper.setMaximum(255)
    self.spinBox_PlusUpper.setSingleStep(1)
    self.spinBox_PlusLower.setMinimum(-255)
    self.spinBox_PlusLower.setMaximum(255)
    self.spinBox_PlusLower.setSingleStep(1)

    self.doubleSpinBox_MultLower.setMinimum(0)
    self.doubleSpinBox_MultLower.setMaximum(999999999)
    self.doubleSpinBox_MultLower.setSingleStep(0.1)
    self.doubleSpinBox_MultUpper.setMinimum(0)
    self.doubleSpinBox_MultUpper.setMaximum(999999999)
    self.doubleSpinBox_MultUpper.setSingleStep(0.1)

    self.doubleSpinBox_GaussianNoiseMean.setMinimum(-255)
    self.doubleSpinBox_GaussianNoiseMean.setMaximum(255)
    self.doubleSpinBox_GaussianNoiseMean.setSingleStep(0.1)

    self.doubleSpinBox_GaussianNoiseScale.setMinimum(0)
    self.doubleSpinBox_GaussianNoiseScale.setMaximum(999999999)
    self.doubleSpinBox_GaussianNoiseScale.setSingleStep(0.1)

    self.spinBox_RefreshAfterNrEpochs.setMinimum(1)
    self.spinBox_RefreshAfterNrEpochs.setMaximum(999999999)
    self.doubleSpinBox_hueDelta.setMaximum(0.5)
    self.doubleSpinBox_hueDelta.setSingleStep(0.01)
    self.doubleSpinBox_contrastHigher.setMaximum(99.9)
    self.doubleSpinBox_contrastHigher.setSingleStep(0.1)
    self.doubleSpinBox_contrastLower.setMaximum(99.9)
    self.doubleSpinBox_contrastLower.setSingleStep(0.1)

    self.doubleSpinBox_saturationLower.setMaximum(99.9)
    self.doubleSpinBox_saturationLower.setSingleStep(0.1)
    self.doubleSpinBox_saturationHigher.setMaximum(99.9)
    self.doubleSpinBox_saturationHigher.setSingleStep(0.1)
    
    self.spinBox_avgBlurMin.setMinimum(0)
    self.spinBox_avgBlurMin.setMaximum(255)
    #self.spinBox_avgBlurMin.setSingleStep(1)
    self.spinBox_avgBlurMax.setMinimum(0)
    self.spinBox_avgBlurMax.setMaximum(255)
    #self.spinBox_avgBlurMax.setSingleStep(0.1)
    
    self.spinBox_gaussBlurMin.setMinimum(0)
    self.spinBox_gaussBlurMin.setMaximum(255)
    #self.spinBox_gaussBlurMin.setSingleStep(0.1)
    self.spinBox_gaussBlurMax.setMinimum(0)
    self.spinBox_gaussBlurMax.setMaximum(255)
    #self.spinBox_gaussBlurMax.setSingleStep(0.1)
                   
    self.tab_ExampleImgs = QtWidgets.QWidget()
    self.tab_ExampleImgs.setObjectName(_fromUtf8("tab_ExampleImgs"))
    self.gridLayout_9 = QtWidgets.QGridLayout(self.tab_ExampleImgs)
    self.gridLayout_9.setObjectName(_fromUtf8("gridLayout_9"))
    self.splitter_4 = QtWidgets.QSplitter(self.tab_ExampleImgs)
    self.splitter_4.setOrientation(QtCore.Qt.Vertical)
    self.splitter_4.setObjectName(_fromUtf8("splitter_4"))
    self.widget2 = QtWidgets.QWidget(self.splitter_4)
    self.widget2.setObjectName(_fromUtf8("widget2"))
    self.horizontalLayout_ExampleImgs = QtWidgets.QHBoxLayout(self.widget2)
    self.horizontalLayout_ExampleImgs.setContentsMargins(0, 0, 0, 0)
    self.horizontalLayout_ExampleImgs.setObjectName(_fromUtf8("horizontalLayout_ExampleImgs"))
    self.comboBox_ShowTrainOrValid = QtWidgets.QComboBox(self.widget2)
    #Insert option for training or valid
    self.comboBox_ShowTrainOrValid.addItems(["Training","Validation"])        
    self.comboBox_ShowTrainOrValid.setObjectName(_fromUtf8("comboBox_ShowTrainOrValid"))
    self.horizontalLayout_ExampleImgs.addWidget(self.comboBox_ShowTrainOrValid)
    self.comboBox_ShowWOrWoAug = QtWidgets.QComboBox(self.widget2)
    self.comboBox_ShowWOrWoAug.addItems(["With Augmentation","Original image"])        
    self.comboBox_ShowWOrWoAug.setObjectName(_fromUtf8("comboBox_ShowWOrWoAug"))
    self.horizontalLayout_ExampleImgs.addWidget(self.comboBox_ShowWOrWoAug)
    self.label_ShowIndex = QtWidgets.QLabel(self.widget2)
    self.label_ShowIndex.setObjectName(_fromUtf8("label_ShowIndex"))
    self.label_ShowIndex.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
    self.horizontalLayout_ExampleImgs.addWidget(self.label_ShowIndex)
    self.spinBox_ShowIndex = QtWidgets.QSpinBox(self.widget2)
    self.spinBox_ShowIndex.setMinimum(0)
    self.spinBox_ShowIndex.setMaximum(9E8)
    self.spinBox_ShowIndex.setObjectName(_fromUtf8("spinBox_ShowIndex"))
    self.horizontalLayout_ExampleImgs.addWidget(self.spinBox_ShowIndex)
    self.pushButton_ShowExamleImgs = QtWidgets.QPushButton(self.widget2)
    self.pushButton_ShowExamleImgs.setObjectName(_fromUtf8("pushButton_ShowExamleImgs"))
    self.pushButton_ShowExamleImgs.clicked.connect(self.action_show_example_imgs)
    self.horizontalLayout_ExampleImgs.addWidget(self.pushButton_ShowExamleImgs)
    self.widget_ViewImages = QtWidgets.QWidget(self.splitter_4)
    self.widget_ViewImages.setObjectName(_fromUtf8("widget_ViewImages"))
    self.gridLayout_9.addWidget(self.splitter_4, 0, 0, 1, 1)
    self.tabWidget_DefineModel.addTab(self.tab_ExampleImgs, _fromUtf8(""))
    self.tab_expert = QtWidgets.QWidget()
    self.tab_expert.setObjectName("tab_expert")
    self.gridLayout_34 = QtWidgets.QGridLayout(self.tab_expert)
    self.gridLayout_34.setObjectName("gridLayout_34")
    self.groupBox_expertMode = QtWidgets.QGroupBox(self.tab_expert)
    self.groupBox_expertMode.setEnabled(True)
    self.groupBox_expertMode.setCheckable(True)
    self.groupBox_expertMode.setChecked(False)
    self.groupBox_expertMode.setObjectName("groupBox_expertMode")
    
    self.gridLayout_35 = QtWidgets.QGridLayout(self.groupBox_expertMode)
    self.gridLayout_35.setObjectName("gridLayout_35")
    self.scrollArea = QtWidgets.QScrollArea(self.groupBox_expertMode)
    self.scrollArea.setWidgetResizable(True)
    self.scrollArea.setObjectName("scrollArea")
    self.scrollAreaWidgetContents = QtWidgets.QWidget()
    self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, -25, 425, 218))
    self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
    self.gridLayout_37 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents)
    self.gridLayout_37.setObjectName("gridLayout_37")
    self.groupBox_modelKerasFit = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
    self.groupBox_modelKerasFit.setObjectName("groupBox_modelKerasFit")
    self.gridLayout_10 = QtWidgets.QGridLayout(self.groupBox_modelKerasFit)
    self.gridLayout_10.setObjectName("gridLayout_10")
    self.label_batchSize = QtWidgets.QLabel(self.groupBox_modelKerasFit)
    self.label_batchSize.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
    self.label_batchSize.setObjectName("label_batchSize")
    self.gridLayout_10.addWidget(self.label_batchSize, 0, 0, 1, 1)
    self.spinBox_batchSize = QtWidgets.QSpinBox(self.groupBox_modelKerasFit)
    self.spinBox_batchSize.setObjectName("spinBox_batchSize")
    self.gridLayout_10.addWidget(self.spinBox_batchSize, 0, 1, 1, 1)
    self.label_epochs = QtWidgets.QLabel(self.groupBox_modelKerasFit)
    self.label_epochs.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
    self.label_epochs.setObjectName("label_epochs")
    self.gridLayout_10.addWidget(self.label_epochs, 0, 2, 1, 1)
    self.spinBox_epochs = QtWidgets.QSpinBox(self.groupBox_modelKerasFit)
    self.spinBox_epochs.setObjectName("spinBox_epochs")
    self.gridLayout_10.addWidget(self.spinBox_epochs, 0, 3, 1, 1)
    self.gridLayout_37.addWidget(self.groupBox_modelKerasFit, 1, 0, 1, 1)
    self.groupBox_lossOptimizer = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
    self.groupBox_lossOptimizer.setObjectName("groupBox_lossOptimizer")
    self.gridLayout_51 = QtWidgets.QGridLayout(self.groupBox_lossOptimizer)
    self.gridLayout_51.setObjectName("gridLayout_51")
    self.checkBox_expt_loss = QtWidgets.QCheckBox(self.groupBox_lossOptimizer)
    self.checkBox_expt_loss.setLayoutDirection(QtCore.Qt.RightToLeft)
    self.checkBox_expt_loss.setObjectName("checkBox_expt_loss")
    self.gridLayout_51.addWidget(self.checkBox_expt_loss, 0, 0, 1, 1)
    self.comboBox_expt_loss = QtWidgets.QComboBox(self.groupBox_lossOptimizer)
    self.comboBox_expt_loss.setEnabled(False)
    self.comboBox_expt_loss.setLayoutDirection(QtCore.Qt.LeftToRight)
    self.comboBox_expt_loss.setObjectName("comboBox_expt_loss")
    self.comboBox_expt_loss.addItem("")
    self.comboBox_expt_loss.addItem("")
    self.comboBox_expt_loss.addItem("")
    self.comboBox_expt_loss.addItem("")
    self.comboBox_expt_loss.addItem("")
    self.comboBox_expt_loss.addItem("")
    self.comboBox_expt_loss.addItem("")
    self.comboBox_expt_loss.addItem("")
    self.comboBox_expt_loss.addItem("")
    self.comboBox_expt_loss.addItem("")
    self.comboBox_expt_loss.addItem("")
    self.comboBox_expt_loss.addItem("")
    self.comboBox_expt_loss.addItem("")
    self.gridLayout_51.addWidget(self.comboBox_expt_loss, 0, 1, 1, 1)
    self.checkBox_optimizer = QtWidgets.QCheckBox(self.groupBox_lossOptimizer)
    self.checkBox_optimizer.setLayoutDirection(QtCore.Qt.RightToLeft)
    self.checkBox_optimizer.setObjectName("checkBox_optimizer")
    self.gridLayout_51.addWidget(self.checkBox_optimizer, 0, 2, 1, 1)
    self.comboBox_optimizer = QtWidgets.QComboBox(self.groupBox_lossOptimizer)
    self.comboBox_optimizer.setEnabled(False)
    self.comboBox_optimizer.setObjectName("comboBox_optimizer")
    self.comboBox_optimizer.addItem("")
    self.comboBox_optimizer.addItem("")
    self.comboBox_optimizer.addItem("")
    self.comboBox_optimizer.addItem("")
    self.comboBox_optimizer.addItem("")
    self.comboBox_optimizer.addItem("")
    self.comboBox_optimizer.addItem("")
    self.gridLayout_51.addWidget(self.comboBox_optimizer, 0, 3, 1, 1)
    self.pushButton_optimizer = QtWidgets.QPushButton(self.groupBox_lossOptimizer)
    self.pushButton_optimizer.setEnabled(False)
    self.pushButton_optimizer.setMaximumSize(QtCore.QSize(40, 16777215))
    self.pushButton_optimizer.setObjectName("pushButton_optimizer")
    self.gridLayout_51.addWidget(self.pushButton_optimizer, 0, 5, 1, 1)

    self.checkBox_lossW = QtWidgets.QCheckBox(self.groupBox_lossOptimizer)
    self.checkBox_lossW.setLayoutDirection(QtCore.Qt.RightToLeft)
    self.checkBox_lossW.setObjectName("checkBox_lossW")
    self.gridLayout_51.addWidget(self.checkBox_lossW, 1, 0, 1, 1)
    self.pushButton_lossW = QtWidgets.QPushButton(self.groupBox_lossOptimizer)
    self.pushButton_lossW.setEnabled(False)
    self.pushButton_lossW.setMaximumSize(QtCore.QSize(40, 16777215))
    self.pushButton_lossW.setObjectName("pushButton_lossW")
    self.pushButton_lossW.setMinimumSize(QtCore.QSize(0, 0))
    self.pushButton_lossW.setMaximumSize(QtCore.QSize(40, 16777215))
    self.gridLayout_51.addWidget(self.pushButton_lossW, 1, 5, 1, 1)
    self.lineEdit_lossW = QtWidgets.QLineEdit(self.groupBox_lossOptimizer)
    self.lineEdit_lossW.setEnabled(False)
    self.lineEdit_lossW.setObjectName("lineEdit_lossW")
    self.gridLayout_51.addWidget(self.lineEdit_lossW, 1, 1, 1, 3)

    self.gridLayout_37.addWidget(self.groupBox_lossOptimizer, 2, 0, 1, 1)
    self.groupBox_learningRate = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
    self.groupBox_learningRate.setEnabled(True)
    self.groupBox_learningRate.setCheckable(True)
    self.groupBox_learningRate.setChecked(False)
    self.groupBox_learningRate.setObjectName("groupBox_learningRate")
    self.gridLayout_50 = QtWidgets.QGridLayout(self.groupBox_learningRate)
    self.gridLayout_50.setObjectName("gridLayout_50")
    self.radioButton_LrCycl = QtWidgets.QRadioButton(self.groupBox_learningRate)
    self.radioButton_LrCycl.setObjectName("radioButton_LrCycl")
    self.gridLayout_50.addWidget(self.radioButton_LrCycl, 1, 0, 1, 1)       

    self.radioButton_LrExpo = QtWidgets.QRadioButton(self.groupBox_learningRate)
    self.radioButton_LrExpo.setObjectName("radioButton_LrExpo")
    self.gridLayout_50.addWidget(self.radioButton_LrExpo, 2, 0, 1, 1)
    self.label_expDecInitLr = QtWidgets.QLabel(self.groupBox_learningRate)
    self.label_expDecInitLr.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
    self.label_expDecInitLr.setObjectName("label_expDecInitLr")
    self.gridLayout_50.addWidget(self.label_expDecInitLr, 2, 1, 1, 1)
    self.doubleSpinBox_expDecInitLr = QtWidgets.QDoubleSpinBox(self.groupBox_learningRate)
    self.doubleSpinBox_expDecInitLr.setEnabled(False)
    self.doubleSpinBox_expDecInitLr.setMaximumSize(QtCore.QSize(63, 16777215))
    self.doubleSpinBox_expDecInitLr.setDecimals(6)
    self.doubleSpinBox_expDecInitLr.setSingleStep(0.0001)
    self.doubleSpinBox_expDecInitLr.setProperty("value", 0.001)
    self.doubleSpinBox_expDecInitLr.setObjectName("doubleSpinBox_expDecInitLr")
    self.gridLayout_50.addWidget(self.doubleSpinBox_expDecInitLr, 2, 2, 1, 1)
    self.label_expDecSteps = QtWidgets.QLabel(self.groupBox_learningRate)
    self.label_expDecSteps.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
    self.label_expDecSteps.setObjectName("label_expDecSteps")
    self.gridLayout_50.addWidget(self.label_expDecSteps, 2, 3, 1, 1)
    self.spinBox_expDecSteps = QtWidgets.QSpinBox(self.groupBox_learningRate)
    self.spinBox_expDecSteps.setEnabled(False)
    self.spinBox_expDecSteps.setMaximumSize(QtCore.QSize(63, 16777215))
    self.spinBox_expDecSteps.setMaximum(999999999)
    self.spinBox_expDecSteps.setProperty("value", 100)
    self.spinBox_expDecSteps.setObjectName("spinBox_expDecSteps")
    self.gridLayout_50.addWidget(self.spinBox_expDecSteps, 2, 4, 1, 1)
    
    self.label_expDecRate = QtWidgets.QLabel(self.groupBox_learningRate)
    self.label_expDecRate.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
    self.label_expDecRate.setObjectName("label_expDecRate")
    self.gridLayout_50.addWidget(self.label_expDecRate, 2, 6, 1, 1)
    self.doubleSpinBox_expDecRate = QtWidgets.QDoubleSpinBox(self.groupBox_learningRate)
    self.doubleSpinBox_expDecRate.setEnabled(False)
    self.doubleSpinBox_expDecRate.setMaximumSize(QtCore.QSize(63, 16777215))
    self.doubleSpinBox_expDecRate.setDecimals(6)
    self.doubleSpinBox_expDecRate.setMaximum(1.0)
    self.doubleSpinBox_expDecRate.setSingleStep(0.01)
    self.doubleSpinBox_expDecRate.setProperty("value", 0.96)
    self.doubleSpinBox_expDecRate.setObjectName("doubleSpinBox_expDecRate")
    self.gridLayout_50.addWidget(self.doubleSpinBox_expDecRate, 2, 7, 1, 1)
    
    self.label_cycLrMin = QtWidgets.QLabel(self.groupBox_learningRate)
    self.label_cycLrMin.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
    self.label_cycLrMin.setObjectName("label_cycLrMin")
    self.gridLayout_50.addWidget(self.label_cycLrMin, 1, 1, 1, 1)
    self.lineEdit_cycLrMin = QtWidgets.QLineEdit(self.groupBox_learningRate)
    self.lineEdit_cycLrMin.setEnabled(False)
    self.lineEdit_cycLrMin.setObjectName("lineEdit_cycLrMin")
    validator = QtGui.QRegExpValidator(QtCore.QRegExp("^[0-9 . , e -]+$")) #validator allows numbers, dots, commas, e and -
    self.lineEdit_cycLrMin.setValidator(validator)
    self.gridLayout_50.addWidget(self.lineEdit_cycLrMin, 1, 2, 1, 1)
    self.lineEdit_cycLrMax = QtWidgets.QLineEdit(self.groupBox_learningRate)
    self.lineEdit_cycLrMax.setEnabled(False)
    self.lineEdit_cycLrMax.setObjectName("lineEdit_cycLrMax")
    self.lineEdit_cycLrMax.setValidator(validator)
    self.gridLayout_50.addWidget(self.lineEdit_cycLrMax, 1, 3, 1, 1)
    self.pushButton_cycLrPopup = QtWidgets.QPushButton(self.groupBox_learningRate)
    self.pushButton_cycLrPopup.setEnabled(False)
    self.pushButton_cycLrPopup.setMaximumSize(QtCore.QSize(50, 16777215))
    self.pushButton_cycLrPopup.setObjectName("pushButton_cycLrPopup")
    self.gridLayout_50.addWidget(self.pushButton_cycLrPopup, 1, 7, 1, 1)
    self.comboBox_cycLrMethod = QtWidgets.QComboBox(self.groupBox_learningRate)
    self.comboBox_cycLrMethod.setEnabled(False)
    self.comboBox_cycLrMethod.setMinimumSize(QtCore.QSize(80, 0))
    self.comboBox_cycLrMethod.setObjectName("comboBox_cycLrMethod")
    self.comboBox_cycLrMethod.addItem("")
    self.comboBox_cycLrMethod.addItem("")
    self.comboBox_cycLrMethod.addItem("")
    self.gridLayout_50.addWidget(self.comboBox_cycLrMethod, 1, 6, 1, 1)
    self.label_cycLrMethod = QtWidgets.QLabel(self.groupBox_learningRate)
    self.label_cycLrMethod.setObjectName("label_cycLrMethod")
    self.label_cycLrMethod.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)

    self.gridLayout_50.addWidget(self.label_cycLrMethod, 1, 4, 1, 1)
    self.radioButton_LrConst = QtWidgets.QRadioButton(self.groupBox_learningRate)
    self.radioButton_LrConst.setChecked(True)
    self.radioButton_LrConst.setObjectName("radioButton_LrConst")
    self.gridLayout_50.addWidget(self.radioButton_LrConst, 0, 0, 1, 1)
    self.doubleSpinBox_learningRate = QtWidgets.QDoubleSpinBox(self.groupBox_learningRate)
    self.doubleSpinBox_learningRate.setEnabled(True)
    self.doubleSpinBox_learningRate.setMaximumSize(QtCore.QSize(63, 16777215))
    self.doubleSpinBox_learningRate.setDecimals(6)
    self.doubleSpinBox_learningRate.setMaximum(999.0)
    self.doubleSpinBox_learningRate.setSingleStep(0.0001)
    self.doubleSpinBox_learningRate.setProperty("value", 0.001)
    self.doubleSpinBox_learningRate.setObjectName("doubleSpinBox_learningRate")
    self.gridLayout_50.addWidget(self.doubleSpinBox_learningRate, 0, 2, 1, 1)
    self.line_2 = QtWidgets.QFrame(self.groupBox_learningRate)
    self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
    self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
    self.line_2.setObjectName("line_2")
    self.gridLayout_50.addWidget(self.line_2, 3, 5, 1, 1)

    self.pushButton_LR_finder = QtWidgets.QPushButton(self.groupBox_learningRate)
    self.pushButton_LR_finder.setObjectName("pushButton_LR_finder")
    self.gridLayout_50.addWidget(self.pushButton_LR_finder, 3, 6, 1, 1)
    self.pushButton_LR_plot = QtWidgets.QPushButton(self.groupBox_learningRate)
    self.pushButton_LR_plot.setObjectName("pushButton_LR_plot")
    self.gridLayout_50.addWidget(self.pushButton_LR_plot, 3, 7, 1, 1)

    self.label_LrConst = QtWidgets.QLabel(self.groupBox_learningRate)
    self.label_LrConst.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
    self.label_LrConst.setObjectName("label_LrConst")
    self.gridLayout_50.addWidget(self.label_LrConst, 0, 1, 1, 1)
    self.gridLayout_37.addWidget(self.groupBox_learningRate, 3, 0, 1, 1)
    
    self.radioButton_LrConst.toggled['bool'].connect(self.doubleSpinBox_learningRate.setEnabled)
    self.radioButton_LrCycl.toggled['bool'].connect(self.lineEdit_cycLrMin.setEnabled)
    self.radioButton_LrCycl.toggled['bool'].connect(self.lineEdit_cycLrMax.setEnabled)
    self.radioButton_LrCycl.toggled['bool'].connect(self.comboBox_cycLrMethod.setEnabled)
    self.radioButton_LrCycl.toggled['bool'].connect(self.pushButton_cycLrPopup.setEnabled)
    self.radioButton_LrExpo.toggled['bool'].connect(self.doubleSpinBox_expDecInitLr.setEnabled)
    self.radioButton_LrExpo.toggled['bool'].connect(self.spinBox_expDecSteps.setEnabled)
    self.radioButton_LrExpo.toggled['bool'].connect(self.doubleSpinBox_expDecRate.setEnabled)
    


    self.groupBox_expertMode.toggled.connect(self.expert_mode_off)
    self.checkBox_expt_loss.stateChanged.connect(self.expert_loss_off)
    self.groupBox_learningRate.toggled.connect(self.expert_learningrate_off)
    self.checkBox_optimizer.stateChanged.connect(self.expert_optimizer_off)
    #optimizer_text = str(self.comboBox_optimizer.currentText())
    self.comboBox_optimizer.currentTextChanged.connect(lambda: self.expert_optimizer_changed(optimizer_text=self.comboBox_optimizer.currentText(),listindex=-1))
    self.doubleSpinBox_learningRate.valueChanged.connect(lambda: self.expert_lr_changed(value=self.doubleSpinBox_learningRate.value(),optimizer_text=self.comboBox_optimizer.currentText(),listindex=-1))





    self.groupBox_regularization = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
    self.groupBox_regularization.setObjectName("groupBox_regularization")
    self.gridLayout_46 = QtWidgets.QGridLayout(self.groupBox_regularization)
    self.gridLayout_46.setObjectName("gridLayout_46")        

    self.horizontalLayout_35 = QtWidgets.QHBoxLayout()
    self.horizontalLayout_35.setObjectName("horizontalLayout_35")
    self.checkBox_trainLastNOnly = QtWidgets.QCheckBox(self.groupBox_regularization)
    self.checkBox_trainLastNOnly.setObjectName("checkBox_trainLastNOnly")
    self.horizontalLayout_35.addWidget(self.checkBox_trainLastNOnly)
    self.spinBox_trainLastNOnly = QtWidgets.QSpinBox(self.groupBox_regularization)
    self.spinBox_trainLastNOnly.setObjectName("spinBox_trainLastNOnly")
    self.horizontalLayout_35.addWidget(self.spinBox_trainLastNOnly)
    self.checkBox_trainDenseOnly = QtWidgets.QCheckBox(self.groupBox_regularization)
    self.checkBox_trainDenseOnly.setObjectName("checkBox_trainDenseOnly")
    self.horizontalLayout_35.addWidget(self.checkBox_trainDenseOnly)
    self.gridLayout_46.addLayout(self.horizontalLayout_35, 3, 0, 1, 1)
    self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
    self.horizontalLayout_2.setObjectName("horizontalLayout_2")
    self.checkBox_dropout = QtWidgets.QCheckBox(self.groupBox_regularization)
    self.checkBox_dropout.setObjectName("checkBox_dropout")
    self.horizontalLayout_2.addWidget(self.checkBox_dropout)
    self.lineEdit_dropout = QtWidgets.QLineEdit(self.groupBox_regularization)
    self.lineEdit_dropout.setObjectName("lineEdit_dropout")
    validator = QtGui.QRegExpValidator(QtCore.QRegExp("^[0-9 . ,]+$")) #validator allows numbers, dots and commas
    #aternatively, I could use "^[0-9 . , \[ \] ]+$" - this would also allow the user to put the brackets. But why? I just do it in the program
    self.lineEdit_dropout.setValidator(validator)        
    self.horizontalLayout_2.addWidget(self.lineEdit_dropout)
    self.gridLayout_46.addLayout(self.horizontalLayout_2, 4, 0, 1, 1)
    self.horizontalLayout_partialTrainability = QtWidgets.QHBoxLayout()
    self.horizontalLayout_partialTrainability.setObjectName("horizontalLayout_partialTrainability")
    self.checkBox_partialTrainability = QtWidgets.QCheckBox(self.groupBox_regularization)
    self.checkBox_partialTrainability.setObjectName("checkBox_partialTrainability")
    self.checkBox_partialTrainability.setEnabled(False)
    self.horizontalLayout_partialTrainability.addWidget(self.checkBox_partialTrainability)
    self.lineEdit_partialTrainability = QtWidgets.QLineEdit(self.groupBox_regularization)
    self.lineEdit_partialTrainability.setEnabled(False)
    self.lineEdit_partialTrainability.setObjectName("lineEdit_partialTrainability")
    self.horizontalLayout_partialTrainability.addWidget(self.lineEdit_partialTrainability)
    self.pushButton_partialTrainability = QtWidgets.QPushButton(self.groupBox_regularization)
    self.pushButton_partialTrainability.setObjectName("pushButton_partialTrainability")
    self.horizontalLayout_partialTrainability.addWidget(self.pushButton_partialTrainability)
    self.gridLayout_46.addLayout(self.horizontalLayout_partialTrainability, 5, 0, 1, 1)
    self.gridLayout_37.addWidget(self.groupBox_regularization, 4, 0, 1, 1)

    self.pushButton_partialTrainability.setEnabled(False)
    self.pushButton_partialTrainability.setMinimumSize(QtCore.QSize(0, 0))
    self.pushButton_partialTrainability.setMaximumSize(QtCore.QSize(40, 16777215))
    self.pushButton_partialTrainability.clicked.connect(self.partialTrainability)
    
    self.groupBox_expertMetrics = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
    self.groupBox_expertMetrics.setObjectName("groupBox_expertMetrics")
    self.gridLayout = QtWidgets.QGridLayout(self.groupBox_expertMetrics)
    self.gridLayout.setObjectName("gridLayout")
    self.checkBox_expertAccuracy = QtWidgets.QCheckBox(self.groupBox_expertMetrics)
    self.checkBox_expertAccuracy.setChecked(True)
    self.checkBox_expertAccuracy.setEnabled(False) #Accuracy is ALWAYS tracked!
    
    self.checkBox_expertAccuracy.setObjectName("checkBox_expertAccuracy")
    self.gridLayout.addWidget(self.checkBox_expertAccuracy, 0, 0, 1, 1)
    self.checkBox_expertF1 = QtWidgets.QCheckBox(self.groupBox_expertMetrics)
    self.checkBox_expertF1.setChecked(False)
    self.checkBox_expertF1.setObjectName("checkBox_expertF1")
    self.gridLayout.addWidget(self.checkBox_expertF1, 0, 1, 1, 1)
    self.checkBox_expertPrecision = QtWidgets.QCheckBox(self.groupBox_expertMetrics)
    self.checkBox_expertPrecision.setObjectName("checkBox_expertPrecision")
    self.gridLayout.addWidget(self.checkBox_expertPrecision, 0, 2, 1, 1)
    self.checkBox_expertRecall = QtWidgets.QCheckBox(self.groupBox_expertMetrics)
    self.checkBox_expertRecall.setObjectName("checkBox_expertRecall")
    self.gridLayout.addWidget(self.checkBox_expertRecall, 0, 3, 1, 1)
    self.gridLayout_37.addWidget(self.groupBox_expertMetrics, 5, 0, 1, 1)

    self.scrollArea.setWidget(self.scrollAreaWidgetContents)
    self.gridLayout_35.addWidget(self.scrollArea, 0, 0, 1, 1)
    self.gridLayout_34.addWidget(self.groupBox_expertMode, 0, 0, 1, 1)
    self.tabWidget_DefineModel.addTab(self.tab_expert, "")

    self.groupBox_Finalize = QtWidgets.QGroupBox(self.splitter_5)
    self.groupBox_Finalize.setObjectName(_fromUtf8("groupBox_Finalize"))
    self.gridLayout_6 = QtWidgets.QGridLayout(self.groupBox_Finalize)
    self.gridLayout_6.setObjectName(_fromUtf8("gridLayout_6"))
    self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
    self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
    self.textBrowser_Info = QtWidgets.QTextBrowser(self.groupBox_Finalize)
    self.textBrowser_Info.setMinimumSize(QtCore.QSize(0, 60))
    self.textBrowser_Info.setMaximumSize(QtCore.QSize(16777215, 500))
    self.textBrowser_Info.setObjectName(_fromUtf8("textBrowser_Info"))
    self.horizontalLayout_3.addWidget(self.textBrowser_Info)
    self.pushButton_FitModel = QtWidgets.QPushButton(self.groupBox_Finalize)
    self.pushButton_FitModel.setMinimumSize(QtCore.QSize(111, 60))
    self.pushButton_FitModel.setMaximumSize(QtCore.QSize(111, 60))
    self.pushButton_FitModel.setObjectName(_fromUtf8("pushButton_FitModel"))
    self.pushButton_FitModel.clicked.connect(lambda: self.action_initialize_model(duties="initialize_train"))
    
    self.horizontalLayout_3.addWidget(self.pushButton_FitModel)
    self.gridLayout_6.addLayout(self.horizontalLayout_3, 0, 0, 1, 1)
    self.gridLayout_17.addWidget(self.splitter_5, 0, 0, 1, 1)
    self.tabWidget_Modelbuilder.addTab(self.tab_Build, _fromUtf8(""))
    self.tab_History = QtWidgets.QWidget()
    sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
    sizePolicy.setHorizontalStretch(0)
    sizePolicy.setVerticalStretch(0)
    sizePolicy.setHeightForWidth(self.tab_History.sizePolicy().hasHeightForWidth())
    self.tab_History.setSizePolicy(sizePolicy)
    self.tab_History.setObjectName(_fromUtf8("tab_History"))
    self.gridLayout_3 = QtWidgets.QGridLayout(self.tab_History)
    self.gridLayout_3.setObjectName(_fromUtf8("gridLayout_3"))
    self.verticalLayout_9 = QtWidgets.QVBoxLayout()
    self.verticalLayout_9.setObjectName(_fromUtf8("verticalLayout_9"))
    self.verticalLayout_HistoryLoad = QtWidgets.QVBoxLayout()
    self.verticalLayout_HistoryLoad.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
    self.verticalLayout_HistoryLoad.setObjectName(_fromUtf8("verticalLayout_HistoryLoad"))
    self.horizontalLayout_HistoryLoad = QtWidgets.QHBoxLayout()
    self.horizontalLayout_HistoryLoad.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
    self.horizontalLayout_HistoryLoad.setObjectName(_fromUtf8("horizontalLayout_HistoryLoad"))
    self.pushButton_Live = QtWidgets.QPushButton(self.tab_History)
    self.pushButton_Live.clicked.connect(self.action_load_history_current)
    self.pushButton_Live.setMinimumSize(QtCore.QSize(93, 28))
    self.pushButton_Live.setMaximumSize(QtCore.QSize(93, 28))
    self.pushButton_Live.setObjectName(_fromUtf8("pushButton_Live"))
    self.horizontalLayout_HistoryLoad.addWidget(self.pushButton_Live)
    self.pushButton_LoadHistory = QtWidgets.QPushButton(self.tab_History)
    self.pushButton_LoadHistory.setMinimumSize(QtCore.QSize(93, 28))
    self.pushButton_LoadHistory.setMaximumSize(QtCore.QSize(93, 28))
    self.pushButton_LoadHistory.clicked.connect(self.action_load_history)
    self.pushButton_LoadHistory.setObjectName(_fromUtf8("pushButton_LoadHistory"))
    self.horizontalLayout_HistoryLoad.addWidget(self.pushButton_LoadHistory)
    self.lineEdit_LoadHistory = QtWidgets.QLineEdit(self.tab_History)
    self.lineEdit_LoadHistory.setDisabled(True)
    self.lineEdit_LoadHistory.setObjectName(_fromUtf8("lineEdit_LoadHistory"))
    self.horizontalLayout_HistoryLoad.addWidget(self.lineEdit_LoadHistory)
    self.verticalLayout_HistoryLoad.addLayout(self.horizontalLayout_HistoryLoad)
    self.horizontalLayout_HistoryLoadInfo = QtWidgets.QHBoxLayout()
    self.horizontalLayout_HistoryLoadInfo.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
    self.horizontalLayout_HistoryLoadInfo.setObjectName(_fromUtf8("horizontalLayout_HistoryLoadInfo"))
    self.tableWidget_HistoryItems = QtWidgets.QTableWidget(self.tab_History)
    self.tableWidget_HistoryItems.setMinimumSize(QtCore.QSize(0, 100))
    self.tableWidget_HistoryItems.setMaximumSize(QtCore.QSize(16777215, 140))
    self.tableWidget_HistoryItems.setObjectName(_fromUtf8("tableWidget_HistoryItems"))
    self.tableWidget_HistoryItems.setColumnCount(7)
    self.tableWidget_HistoryItems.setRowCount(0)
    self.horizontalLayout_HistoryLoadInfo.addWidget(self.tableWidget_HistoryItems)

    self.verticalLayout_UpdatePlot = QtWidgets.QVBoxLayout()
    self.pushButton_UpdateHistoryPlot = QtWidgets.QPushButton(self.tab_History)
    self.pushButton_UpdateHistoryPlot.clicked.connect(self.update_historyplot)
    sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
    sizePolicy.setHorizontalStretch(0)
    sizePolicy.setVerticalStretch(0)
    sizePolicy.setHeightForWidth(self.pushButton_UpdateHistoryPlot.sizePolicy().hasHeightForWidth())
    self.pushButton_UpdateHistoryPlot.setSizePolicy(sizePolicy)
    self.pushButton_UpdateHistoryPlot.setObjectName(_fromUtf8("pushButton_UpdateHistoryPlot"))
    self.verticalLayout_UpdatePlot.addWidget(self.pushButton_UpdateHistoryPlot)

    self.horizontalLayout_rollmedi = QtWidgets.QHBoxLayout()
    self.checkBox_rollingMedian = QtWidgets.QCheckBox(self.tab_History)
    self.checkBox_rollingMedian.setMinimumSize(QtCore.QSize(100, 19))
    self.checkBox_rollingMedian.setMaximumSize(QtCore.QSize(125, 25))
    self.checkBox_rollingMedian.toggled.connect(self.checkBox_rollingMedian_statechange)
    
    self.checkBox_rollingMedian.setObjectName(_fromUtf8("checkBox_rollingMedian"))
    self.horizontalLayout_rollmedi.addWidget(self.checkBox_rollingMedian)        
    self.horizontalSlider_rollmedi = QtWidgets.QSlider(self.tab_History)
    self.horizontalSlider_rollmedi.setOrientation(QtCore.Qt.Horizontal)
    self.horizontalSlider_rollmedi.setMinimumSize(QtCore.QSize(50, 19))
    self.horizontalSlider_rollmedi.setMaximumSize(QtCore.QSize(50, 25))
    #Adjust the horizontalSlider_rollmedi
    self.horizontalSlider_rollmedi.setSingleStep(1)
    self.horizontalSlider_rollmedi.setMinimum(1)
    self.horizontalSlider_rollmedi.setMaximum(50)
    self.horizontalSlider_rollmedi.setValue(10)
    self.horizontalSlider_rollmedi.setEnabled(False)
    
    self.horizontalSlider_rollmedi.setObjectName(_fromUtf8("horizontalSlider_rollmedi"))
    self.horizontalLayout_rollmedi.addWidget(self.horizontalSlider_rollmedi)
    self.verticalLayout_UpdatePlot.addLayout(self.horizontalLayout_rollmedi)

    self.checkBox_linearFit = QtWidgets.QCheckBox(self.tab_History)
    self.checkBox_linearFit.setObjectName(_fromUtf8("checkBox_linearFit"))
    self.verticalLayout_UpdatePlot.addWidget(self.checkBox_linearFit)
    self.horizontalLayout_HistoryLoadInfo.addLayout(self.verticalLayout_UpdatePlot)
    
    self.verticalLayout_HistoryLoad.addLayout(self.horizontalLayout_HistoryLoadInfo)
    self.verticalLayout_9.addLayout(self.verticalLayout_HistoryLoad)
    self.widget_Scatterplot = pg.GraphicsLayoutWidget(self.tab_History)
    sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
    sizePolicy.setHorizontalStretch(0)
    sizePolicy.setVerticalStretch(0)
    sizePolicy.setHeightForWidth(self.widget_Scatterplot.sizePolicy().hasHeightForWidth())
    self.widget_Scatterplot.setSizePolicy(sizePolicy)
    self.widget_Scatterplot.setMinimumSize(QtCore.QSize(491, 350))
    self.widget_Scatterplot.setObjectName(_fromUtf8("widget_Scatterplot"))
    self.verticalLayout_9.addWidget(self.widget_Scatterplot)
    self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
    self.horizontalLayout_7.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
    self.horizontalLayout_7.setObjectName(_fromUtf8("horizontalLayout_7"))
    self.verticalLayout_convert = QtWidgets.QVBoxLayout()
    self.verticalLayout_convert.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
    self.verticalLayout_convert.setObjectName(_fromUtf8("verticalLayout_convert"))

    self.verticalLayout_4 = QtWidgets.QVBoxLayout()
    self.verticalLayout_4.setObjectName(_fromUtf8("verticalLayout_4"))
    self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
    self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
    self.combobox_initial_format = QtWidgets.QComboBox(self.tab_History)
    self.combobox_initial_format.setObjectName(_fromUtf8("combobox_initial_format"))
    self.horizontalLayout_2.addWidget(self.combobox_initial_format)
    self.verticalLayout_4.addLayout(self.horizontalLayout_2)
    self.pushButton_LoadModel = QtWidgets.QPushButton(self.tab_History)
    self.pushButton_LoadModel.setMinimumSize(QtCore.QSize(123, 61))
    self.pushButton_LoadModel.setMaximumSize(QtCore.QSize(150, 61))
    self.pushButton_LoadModel.setObjectName(_fromUtf8("pushButton_LoadModel"))
    self.pushButton_LoadModel.clicked.connect(self.history_tab_get_model_path)
    self.verticalLayout_4.addWidget(self.pushButton_LoadModel)
    self.horizontalLayout_7.addLayout(self.verticalLayout_4)
    self.textBrowser_SelectedModelInfo = QtWidgets.QTextBrowser(self.tab_History)
    self.textBrowser_SelectedModelInfo.setMinimumSize(QtCore.QSize(0, 120))
    self.textBrowser_SelectedModelInfo.setMaximumSize(QtCore.QSize(16777215, 120))
    self.textBrowser_SelectedModelInfo.setObjectName(_fromUtf8("textBrowser_SelectedModelInfo"))
    self.horizontalLayout_7.addWidget(self.textBrowser_SelectedModelInfo)
    
    self.comboBox_convertTo = QtWidgets.QComboBox(self.tab_History)
    self.comboBox_convertTo.setObjectName(_fromUtf8("comboBox_convertTo"))
    self.verticalLayout_convert.addWidget(self.comboBox_convertTo)

    self.pushButton_convertModel = QtWidgets.QPushButton(self.tab_History)
    self.pushButton_convertModel.setMinimumSize(QtCore.QSize(0, 61))
    self.pushButton_convertModel.setMaximumSize(QtCore.QSize(16777215, 61))
    self.pushButton_convertModel.setObjectName(_fromUtf8("pushButton_convertModel"))
    self.pushButton_convertModel.setEnabled(True)
    self.pushButton_convertModel.clicked.connect(self.history_tab_convertModel)
    self.verticalLayout_convert.addWidget(self.pushButton_convertModel)
    
    self.horizontalLayout_7.addLayout(self.verticalLayout_convert)
            
    self.verticalLayout_9.addLayout(self.horizontalLayout_7)
    self.gridLayout_3.addLayout(self.verticalLayout_9, 0, 0, 1, 1)
    self.tabWidget_Modelbuilder.addTab(self.tab_History, _fromUtf8(""))


    ############Icon####################
    self.pushButton_FitModel.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"continue.png")))

    ############Icons Expert tab########
    #self.pushButton_LR_finder.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"lr_screen.png")))
    icon = QtGui.QIcon()
    icon.addPixmap(QtGui.QPixmap(os.path.join(dir_root,"art",Default_dict["Icon theme"],"lr_screen.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
    self.pushButton_LR_finder.setIcon(icon)
    icon = QtGui.QIcon()
    icon.addPixmap(QtGui.QPixmap(os.path.join(dir_root,"art",Default_dict["Icon theme"],"lr_plot.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
    self.pushButton_LR_plot.setIcon(icon)

    icon = QtGui.QIcon()
    icon.addPixmap(QtGui.QPixmap(os.path.join(dir_root,"art",Default_dict["Icon theme"],"lr_const.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
    self.radioButton_LrConst.setIcon(icon)
    icon = QtGui.QIcon()
    icon.addPixmap(QtGui.QPixmap(os.path.join(dir_root,"art",Default_dict["Icon theme"],"lr_cycle.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
    self.radioButton_LrCycl.setIcon(icon)
    icon = QtGui.QIcon()
    icon.addPixmap(QtGui.QPixmap(os.path.join(dir_root,"art",Default_dict["Icon theme"],"lr_exponential.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
    self.radioButton_LrExpo.setIcon(icon)








    #########################Assess Model tab##############################


    self.tab_AssessModel = QtWidgets.QWidget()
    self.tab_AssessModel.setObjectName("tab_AssessModel")
    self.gridLayout_23 = QtWidgets.QGridLayout(self.tab_AssessModel)
    self.gridLayout_23.setObjectName("gridLayout_23")
    self.splitter_7 = QtWidgets.QSplitter(self.tab_AssessModel)
    self.splitter_7.setOrientation(QtCore.Qt.Vertical)
    self.splitter_7.setObjectName("splitter_7")
    self.widget = QtWidgets.QWidget(self.splitter_7)
    self.widget.setObjectName("widget")
    self.verticalLayout_10 = QtWidgets.QVBoxLayout(self.widget)
    self.verticalLayout_10.setContentsMargins(0, 0, 0, 0)
    self.verticalLayout_10.setObjectName("verticalLayout_10")
    self.groupBox_loadModel = QtWidgets.QGroupBox(self.widget)
    sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
    sizePolicy.setHorizontalStretch(0)
    sizePolicy.setVerticalStretch(0)
    sizePolicy.setHeightForWidth(self.groupBox_loadModel.sizePolicy().hasHeightForWidth())
    self.groupBox_loadModel.setSizePolicy(sizePolicy)
    self.groupBox_loadModel.setMinimumSize(QtCore.QSize(0, 101))
    self.groupBox_loadModel.setMaximumSize(QtCore.QSize(16777215, 101))
    self.groupBox_loadModel.setObjectName("groupBox_loadModel")
    self.gridLayout_4 = QtWidgets.QGridLayout(self.groupBox_loadModel)
    self.gridLayout_4.setObjectName("gridLayout_4")
    self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
    self.horizontalLayout_10.setObjectName("horizontalLayout_10")
    self.pushButton_LoadModel_2 = QtWidgets.QPushButton(self.groupBox_loadModel)
    self.pushButton_LoadModel_2.setMinimumSize(QtCore.QSize(123, 24))
    self.pushButton_LoadModel_2.setMaximumSize(QtCore.QSize(123, 24))
    self.pushButton_LoadModel_2.setObjectName("pushButton_LoadModel_2")
    self.horizontalLayout_10.addWidget(self.pushButton_LoadModel_2)
    self.lineEdit_LoadModel_2 = QtWidgets.QLineEdit(self.groupBox_loadModel)
    self.lineEdit_LoadModel_2.setEnabled(False)
    self.lineEdit_LoadModel_2.setObjectName("lineEdit_LoadModel_2")
    self.horizontalLayout_10.addWidget(self.lineEdit_LoadModel_2)
    self.comboBox_loadedRGBorGray = QtWidgets.QComboBox(self.groupBox_loadModel)
    self.comboBox_loadedRGBorGray.setEnabled(False)
    self.comboBox_loadedRGBorGray.setObjectName("comboBox_loadedRGBorGray")
    self.horizontalLayout_10.addWidget(self.comboBox_loadedRGBorGray)
    self.gridLayout_4.addLayout(self.horizontalLayout_10, 0, 0, 1, 1)
    self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
    self.horizontalLayout_11.setObjectName("horizontalLayout_11")
    self.label_ModelIndex_2 = QtWidgets.QLabel(self.groupBox_loadModel)
    sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
    sizePolicy.setHorizontalStretch(0)
    sizePolicy.setVerticalStretch(0)
    sizePolicy.setHeightForWidth(self.label_ModelIndex_2.sizePolicy().hasHeightForWidth())
    self.label_ModelIndex_2.setSizePolicy(sizePolicy)
    self.label_ModelIndex_2.setMinimumSize(QtCore.QSize(68, 25))
    self.label_ModelIndex_2.setMaximumSize(QtCore.QSize(68, 25))
    self.label_ModelIndex_2.setObjectName("label_ModelIndex_2")
    self.horizontalLayout_11.addWidget(self.label_ModelIndex_2)
    self.spinBox_ModelIndex_2 = QtWidgets.QSpinBox(self.groupBox_loadModel)
    self.spinBox_ModelIndex_2.setEnabled(False)
    self.spinBox_ModelIndex_2.setMinimumSize(QtCore.QSize(46, 22))
    self.spinBox_ModelIndex_2.setMaximumSize(QtCore.QSize(46, 22))
    self.spinBox_ModelIndex_2.setObjectName("spinBox_ModelIndex_2")
    self.horizontalLayout_11.addWidget(self.spinBox_ModelIndex_2)
    self.lineEdit_ModelSelection_2 = QtWidgets.QLineEdit(self.groupBox_loadModel)
    self.lineEdit_ModelSelection_2.setEnabled(False)
    self.lineEdit_ModelSelection_2.setObjectName("lineEdit_ModelSelection_2")
    self.horizontalLayout_11.addWidget(self.lineEdit_ModelSelection_2)
    self.label_Normalization_2 = QtWidgets.QLabel(self.groupBox_loadModel)
    self.label_Normalization_2.setObjectName("label_Normalization_2")
    self.horizontalLayout_11.addWidget(self.label_Normalization_2)
    self.comboBox_Normalization_2 = QtWidgets.QComboBox(self.groupBox_loadModel)
    self.comboBox_Normalization_2.setEnabled(False)
    self.comboBox_Normalization_2.setObjectName("comboBox_Normalization_2")
    self.horizontalLayout_11.addWidget(self.comboBox_Normalization_2)
    self.label_Crop_2 = QtWidgets.QLabel(self.groupBox_loadModel)
    self.label_Crop_2.setObjectName("label_Crop_2")
    self.horizontalLayout_11.addWidget(self.label_Crop_2)
    self.spinBox_Crop_2 = QtWidgets.QSpinBox(self.groupBox_loadModel)
    self.spinBox_Crop_2.setEnabled(False)
    self.spinBox_Crop_2.setObjectName("spinBox_Crop_2")
    self.horizontalLayout_11.addWidget(self.spinBox_Crop_2)
    self.label_OutClasses_2 = QtWidgets.QLabel(self.groupBox_loadModel)
    self.label_OutClasses_2.setObjectName("label_OutClasses_2")
    self.horizontalLayout_11.addWidget(self.label_OutClasses_2)
    self.spinBox_OutClasses_2 = QtWidgets.QSpinBox(self.groupBox_loadModel)
    self.spinBox_OutClasses_2.setEnabled(False)
    self.spinBox_OutClasses_2.setObjectName("spinBox_OutClasses_2")
    self.horizontalLayout_11.addWidget(self.spinBox_OutClasses_2)
    self.gridLayout_4.addLayout(self.horizontalLayout_11, 1, 0, 1, 1)
    self.verticalLayout_10.addWidget(self.groupBox_loadModel)
    self.horizontalLayout_16 = QtWidgets.QHBoxLayout()
    self.horizontalLayout_16.setObjectName("horizontalLayout_16")
    self.groupBox_validData = QtWidgets.QGroupBox(self.widget)
    self.groupBox_validData.setMaximumSize(QtCore.QSize(150, 250))
    self.groupBox_validData.setObjectName("groupBox_validData")
    self.gridLayout_14 = QtWidgets.QGridLayout(self.groupBox_validData)
    self.gridLayout_14.setObjectName("gridLayout_14")
    self.horizontalLayout_17 = QtWidgets.QHBoxLayout()
    self.horizontalLayout_17.setObjectName("horizontalLayout_17")
    self.pushButton_ImportValidFromNpy = QtWidgets.QPushButton(self.groupBox_validData)
    self.pushButton_ImportValidFromNpy.setMinimumSize(QtCore.QSize(65, 28))
    self.pushButton_ImportValidFromNpy.setObjectName("pushButton_ImportValidFromNpy")
    self.horizontalLayout_17.addWidget(self.pushButton_ImportValidFromNpy)
    self.pushButton_ExportValidToNpy = QtWidgets.QPushButton(self.groupBox_validData)
    self.pushButton_ExportValidToNpy.setMinimumSize(QtCore.QSize(55, 0))
    self.pushButton_ExportValidToNpy.setObjectName("pushButton_ExportValidToNpy")
    self.horizontalLayout_17.addWidget(self.pushButton_ExportValidToNpy)
    self.gridLayout_14.addLayout(self.horizontalLayout_17, 1, 0, 1, 1)
    self.tableWidget_Info_2 = QtWidgets.QTableWidget(self.groupBox_validData)
    self.tableWidget_Info_2.setObjectName("tableWidget_Info_2")
    self.tableWidget_Info_2.setColumnCount(0)
    self.tableWidget_Info_2.setRowCount(0)
    self.gridLayout_14.addWidget(self.tableWidget_Info_2, 0, 0, 1, 1)
    self.horizontalLayout_16.addWidget(self.groupBox_validData)
    self.verticalLayout_9 = QtWidgets.QVBoxLayout()
    self.verticalLayout_9.setObjectName("verticalLayout_9")
    self.groupBox_InferenceTime = QtWidgets.QGroupBox(self.widget)
    self.groupBox_InferenceTime.setMinimumSize(QtCore.QSize(0, 71))
    self.groupBox_InferenceTime.setMaximumSize(QtCore.QSize(16777215, 71))
    self.groupBox_InferenceTime.setObjectName("groupBox_InferenceTime")
    self.gridLayout_20 = QtWidgets.QGridLayout(self.groupBox_InferenceTime)
    self.gridLayout_20.setObjectName("gridLayout_20")
    self.lineEdit_InferenceTime = QtWidgets.QLineEdit(self.groupBox_InferenceTime)
    self.lineEdit_InferenceTime.setObjectName("lineEdit_InferenceTime")
    self.gridLayout_20.addWidget(self.lineEdit_InferenceTime, 0, 2, 1, 1)
    self.pushButton_CompInfTime = QtWidgets.QPushButton(self.groupBox_InferenceTime)
    self.pushButton_CompInfTime.setMinimumSize(QtCore.QSize(121, 31))
    self.pushButton_CompInfTime.setMaximumSize(QtCore.QSize(121, 31))
    self.pushButton_CompInfTime.setObjectName("pushButton_CompInfTime")
    self.gridLayout_20.addWidget(self.pushButton_CompInfTime, 0, 0, 1, 1)
    self.spinBox_inftime_nr_images = QtWidgets.QSpinBox(self.groupBox_InferenceTime)
    self.spinBox_inftime_nr_images.setObjectName("spinBox_inftime_nr_images")
    self.gridLayout_20.addWidget(self.spinBox_inftime_nr_images, 0, 1, 1, 1)
    self.verticalLayout_9.addWidget(self.groupBox_InferenceTime)
    
    
    self.groupBox_classify = QtWidgets.QGroupBox(self.widget)
    self.groupBox_classify.setObjectName("groupBox_classify")
    self.gridLayout_36 = QtWidgets.QGridLayout(self.groupBox_classify)
    self.gridLayout_36.setObjectName("gridLayout_36")
    self.comboBox_scoresOrPrediction = QtWidgets.QComboBox(self.groupBox_classify)
    self.comboBox_scoresOrPrediction.setObjectName("comboBox_scoresOrPrediction")
    self.gridLayout_36.addWidget(self.comboBox_scoresOrPrediction, 0, 1, 1, 1)
    self.pushButton_classify = QtWidgets.QPushButton(self.groupBox_classify)
    self.pushButton_classify.setObjectName("pushButton_classify")
    self.gridLayout_36.addWidget(self.pushButton_classify, 0, 2, 1, 1)
    self.horizontalLayout_36 = QtWidgets.QHBoxLayout()
    self.horizontalLayout_36.setObjectName("horizontalLayout_36")
    self.radioButton_selectAll = QtWidgets.QRadioButton(self.groupBox_classify)
    self.radioButton_selectAll.setObjectName("radioButton_selectAll")
    self.horizontalLayout_36.addWidget(self.radioButton_selectAll)
    self.radioButton_selectDataSet = QtWidgets.QRadioButton(self.groupBox_classify)
    self.radioButton_selectDataSet.setMinimumSize(QtCore.QSize(10, 22))
    self.radioButton_selectDataSet.setMaximumSize(QtCore.QSize(22, 16))
    self.radioButton_selectDataSet.setText("")
    self.radioButton_selectDataSet.setObjectName("radioButton_selectDataSet")
    self.horizontalLayout_36.addWidget(self.radioButton_selectDataSet)
    self.comboBox_selectData = QtWidgets.QComboBox(self.groupBox_classify)
    self.comboBox_selectData.setObjectName("comboBox_selectData")
    self.horizontalLayout_36.addWidget(self.comboBox_selectData)
    self.gridLayout_36.addLayout(self.horizontalLayout_36, 0, 0, 1, 1)
    self.verticalLayout_9.addWidget(self.groupBox_classify)
    
    self.horizontalLayout = QtWidgets.QHBoxLayout()
    self.horizontalLayout.setObjectName("horizontalLayout")
    self.groupBox_settings = QtWidgets.QGroupBox(self.widget)
    self.groupBox_settings.setMinimumSize(QtCore.QSize(0, 99))
    self.groupBox_settings.setMaximumSize(QtCore.QSize(16777215, 99))
    self.groupBox_settings.setObjectName("groupBox_settings")
    self.gridLayout_22 = QtWidgets.QGridLayout(self.groupBox_settings)
    self.gridLayout_22.setObjectName("gridLayout_22")
    self.horizontalLayout_AssessModelSettings = QtWidgets.QHBoxLayout()
    self.horizontalLayout_AssessModelSettings.setObjectName("horizontalLayout_AssessModelSettings")
    self.verticalLayout_AssessModelSettings = QtWidgets.QVBoxLayout()
    self.verticalLayout_AssessModelSettings.setObjectName("verticalLayout_AssessModelSettings")
    self.horizontalLayout_AssessModelSettings_2 = QtWidgets.QHBoxLayout()
    self.horizontalLayout_AssessModelSettings_2.setObjectName("horizontalLayout_AssessModelSettings_2")
    self.label_SortingIndex = QtWidgets.QLabel(self.groupBox_settings)
    self.label_SortingIndex.setObjectName("label_SortingIndex")
    self.horizontalLayout_AssessModelSettings_2.addWidget(self.label_SortingIndex)
    self.spinBox_indexOfInterest = QtWidgets.QSpinBox(self.groupBox_settings)
    self.spinBox_indexOfInterest.setObjectName("spinBox_indexOfInterest")
    self.horizontalLayout_AssessModelSettings_2.addWidget(self.spinBox_indexOfInterest)
    self.verticalLayout_AssessModelSettings.addLayout(self.horizontalLayout_AssessModelSettings_2)
    self.horizontalLayout_AssessModelSettings_3 = QtWidgets.QHBoxLayout()
    self.horizontalLayout_AssessModelSettings_3.setObjectName("horizontalLayout_AssessModelSettings_3")
    self.checkBox_SortingThresh = QtWidgets.QCheckBox(self.groupBox_settings)
    self.checkBox_SortingThresh.setObjectName("checkBox_SortingThresh")
    self.horizontalLayout_AssessModelSettings_3.addWidget(self.checkBox_SortingThresh)
    self.doubleSpinBox_sortingThresh = QtWidgets.QDoubleSpinBox(self.groupBox_settings)
    self.doubleSpinBox_sortingThresh.setObjectName("doubleSpinBox_sortingThresh")
    self.horizontalLayout_AssessModelSettings_3.addWidget(self.doubleSpinBox_sortingThresh)
    self.verticalLayout_AssessModelSettings.addLayout(self.horizontalLayout_AssessModelSettings_3)
    self.horizontalLayout_AssessModelSettings.addLayout(self.verticalLayout_AssessModelSettings)
    self.verticalLayout_13 = QtWidgets.QVBoxLayout()
    self.verticalLayout_13.setObjectName("verticalLayout_13")
    self.pushButton_AssessModel = QtWidgets.QPushButton(self.groupBox_settings)
    self.pushButton_AssessModel.setMinimumSize(QtCore.QSize(0, 36))
    self.pushButton_AssessModel.setMaximumSize(QtCore.QSize(16777215, 16777215))
    font = QtGui.QFont()
    font.setBold(True)
    font.setWeight(75)
    self.pushButton_AssessModel.setFont(font)
    self.pushButton_AssessModel.setAutoFillBackground(False)
    self.pushButton_AssessModel.setShortcut("")
    self.pushButton_AssessModel.setCheckable(False)
    self.pushButton_AssessModel.setChecked(False)
    self.pushButton_AssessModel.setAutoDefault(False)
    self.pushButton_AssessModel.setDefault(False)
    self.pushButton_AssessModel.setFlat(False)
    self.pushButton_AssessModel.setObjectName("pushButton_AssessModel")
    self.verticalLayout_13.addWidget(self.pushButton_AssessModel)
    self.comboBox_probability_histogram = QtWidgets.QComboBox(self.groupBox_settings)
    self.comboBox_probability_histogram.setMaximumSize(QtCore.QSize(16777215, 18))
    self.comboBox_probability_histogram.setObjectName("comboBox_probability_histogram")
    self.verticalLayout_13.addWidget(self.comboBox_probability_histogram)
    self.horizontalLayout_AssessModelSettings.addLayout(self.verticalLayout_13)
    self.gridLayout_22.addLayout(self.horizontalLayout_AssessModelSettings, 0, 0, 1, 1)
    self.horizontalLayout.addWidget(self.groupBox_settings)
    self.groupBox_3rdPlotSettings = QtWidgets.QGroupBox(self.widget)
    self.groupBox_3rdPlotSettings.setMinimumSize(QtCore.QSize(0, 91))
    self.groupBox_3rdPlotSettings.setMaximumSize(QtCore.QSize(16777215, 91))
    self.groupBox_3rdPlotSettings.setObjectName("groupBox_3rdPlotSettings")
    self.gridLayout_21 = QtWidgets.QGridLayout(self.groupBox_3rdPlotSettings)
    self.gridLayout_21.setObjectName("gridLayout_21")
    self.horizontalLayout_20 = QtWidgets.QHBoxLayout()
    self.horizontalLayout_20.setObjectName("horizontalLayout_20")
    self.verticalLayout_3rdPlotSettings = QtWidgets.QVBoxLayout()
    self.verticalLayout_3rdPlotSettings.setObjectName("verticalLayout_3rdPlotSettings")
    self.label_3rdPlot = QtWidgets.QLabel(self.groupBox_3rdPlotSettings)
    self.label_3rdPlot.setObjectName("label_3rdPlot")
    self.verticalLayout_3rdPlotSettings.addWidget(self.label_3rdPlot)
    self.comboBox_3rdPlot = QtWidgets.QComboBox(self.groupBox_3rdPlotSettings)
    self.comboBox_3rdPlot.setObjectName("comboBox_3rdPlot")
    self.verticalLayout_3rdPlotSettings.addWidget(self.comboBox_3rdPlot)
    self.horizontalLayout_20.addLayout(self.verticalLayout_3rdPlotSettings)
    self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
    self.horizontalLayout_12.setObjectName("horizontalLayout_12")
    self.verticalLayout_3rdPlotSettings_2 = QtWidgets.QVBoxLayout()
    self.verticalLayout_3rdPlotSettings_2.setObjectName("verticalLayout_3rdPlotSettings_2")
    self.label_Indx1 = QtWidgets.QLabel(self.groupBox_3rdPlotSettings)
    self.label_Indx1.setObjectName("label_Indx1")
    self.verticalLayout_3rdPlotSettings_2.addWidget(self.label_Indx1)
    self.spinBox_Indx1 = QtWidgets.QSpinBox(self.groupBox_3rdPlotSettings)
    self.spinBox_Indx1.setEnabled(False)
    self.spinBox_Indx1.setObjectName("spinBox_Indx1")
    self.verticalLayout_3rdPlotSettings_2.addWidget(self.spinBox_Indx1)
    self.horizontalLayout_12.addLayout(self.verticalLayout_3rdPlotSettings_2)
    self.verticalLayout_3rdPlotSettings_3 = QtWidgets.QVBoxLayout()
    self.verticalLayout_3rdPlotSettings_3.setObjectName("verticalLayout_3rdPlotSettings_3")
    self.label_Indx2 = QtWidgets.QLabel(self.groupBox_3rdPlotSettings)
    self.label_Indx2.setObjectName("label_Indx2")
    self.verticalLayout_3rdPlotSettings_3.addWidget(self.label_Indx2)
    self.spinBox_Indx2 = QtWidgets.QSpinBox(self.groupBox_3rdPlotSettings)
    self.spinBox_Indx2.setEnabled(False)
    self.spinBox_Indx2.setObjectName("spinBox_Indx2")
    self.verticalLayout_3rdPlotSettings_3.addWidget(self.spinBox_Indx2)
    self.horizontalLayout_12.addLayout(self.verticalLayout_3rdPlotSettings_3)
    self.horizontalLayout_20.addLayout(self.horizontalLayout_12)
    self.gridLayout_21.addLayout(self.horizontalLayout_20, 0, 0, 1, 1)
    self.horizontalLayout.addWidget(self.groupBox_3rdPlotSettings)
    self.verticalLayout_9.addLayout(self.horizontalLayout)
    self.horizontalLayout_16.addLayout(self.verticalLayout_9)
    self.verticalLayout_10.addLayout(self.horizontalLayout_16)
    self.splitter_8 = QtWidgets.QSplitter(self.splitter_7)
    self.splitter_8.setOrientation(QtCore.Qt.Horizontal)
    self.splitter_8.setObjectName("splitter_8")
    self.groupBox_confusionMatrixPlot = QtWidgets.QGroupBox(self.splitter_8)
    sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
    sizePolicy.setHorizontalStretch(0)
    sizePolicy.setVerticalStretch(0)
    sizePolicy.setHeightForWidth(self.groupBox_confusionMatrixPlot.sizePolicy().hasHeightForWidth())
    self.groupBox_confusionMatrixPlot.setSizePolicy(sizePolicy)
    self.groupBox_confusionMatrixPlot.setBaseSize(QtCore.QSize(250, 500))
    self.groupBox_confusionMatrixPlot.setObjectName("groupBox_confusionMatrixPlot")
    self.gridLayout_16 = QtWidgets.QGridLayout(self.groupBox_confusionMatrixPlot)
    self.gridLayout_16.setObjectName("gridLayout_16")
    self.horizontalLayout_24 = QtWidgets.QHBoxLayout()
    self.horizontalLayout_24.setObjectName("horizontalLayout_24")
    self.verticalLayout_11 = QtWidgets.QVBoxLayout()
    self.verticalLayout_11.setObjectName("verticalLayout_11")
    self.horizontalLayout_21 = QtWidgets.QHBoxLayout()
    self.horizontalLayout_21.setObjectName("horizontalLayout_21")
    self.label_True_CM1 = QtWidgets.QLabel(self.groupBox_confusionMatrixPlot)
    font = QtGui.QFont()
    font.setBold(True)
    font.setWeight(75)
    font.setStrikeOut(False)
    font.setKerning(True)
    font.setStyleStrategy(QtGui.QFont.PreferDefault)
    self.label_True_CM1.setFont(font)
    self.label_True_CM1.setObjectName("label_True_CM1")
    self.horizontalLayout_21.addWidget(self.label_True_CM1)
    self.tableWidget_CM1 = QtWidgets.QTableWidget(self.groupBox_confusionMatrixPlot)
    sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
    sizePolicy.setHorizontalStretch(0)
    sizePolicy.setVerticalStretch(0)
    sizePolicy.setHeightForWidth(self.tableWidget_CM1.sizePolicy().hasHeightForWidth())
    self.tableWidget_CM1.setSizePolicy(sizePolicy)
    self.tableWidget_CM1.setBaseSize(QtCore.QSize(250, 250))
    self.tableWidget_CM1.setObjectName("tableWidget_CM1")
    self.tableWidget_CM1.setColumnCount(0)
    self.tableWidget_CM1.setRowCount(0)
    self.horizontalLayout_21.addWidget(self.tableWidget_CM1)
    self.verticalLayout_11.addLayout(self.horizontalLayout_21)
    self.horizontalLayout_18 = QtWidgets.QHBoxLayout()
    self.horizontalLayout_18.setObjectName("horizontalLayout_18")
    self.pushButton_CM1_to_Clipboard = QtWidgets.QPushButton(self.groupBox_confusionMatrixPlot)
    self.pushButton_CM1_to_Clipboard.setObjectName("pushButton_CM1_to_Clipboard")
    self.horizontalLayout_18.addWidget(self.pushButton_CM1_to_Clipboard)
    self.label_Pred_CM1 = QtWidgets.QLabel(self.groupBox_confusionMatrixPlot)
    font = QtGui.QFont()
    font.setPointSize(8)
    font.setBold(True)
    font.setWeight(75)
    self.label_Pred_CM1.setFont(font)
    self.label_Pred_CM1.setLayoutDirection(QtCore.Qt.LeftToRight)
    self.label_Pred_CM1.setAutoFillBackground(False)
    self.label_Pred_CM1.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
    self.label_Pred_CM1.setObjectName("label_Pred_CM1")
    self.horizontalLayout_18.addWidget(self.label_Pred_CM1)
    self.verticalLayout_11.addLayout(self.horizontalLayout_18)
    self.horizontalLayout_24.addLayout(self.verticalLayout_11)
    self.verticalLayout_12 = QtWidgets.QVBoxLayout()
    self.verticalLayout_12.setObjectName("verticalLayout_12")
    self.horizontalLayout_23 = QtWidgets.QHBoxLayout()
    self.horizontalLayout_23.setObjectName("horizontalLayout_23")
    self.label_True_CM2 = QtWidgets.QLabel(self.groupBox_confusionMatrixPlot)
    font = QtGui.QFont()
    font.setBold(True)
    font.setWeight(75)
    font.setStrikeOut(False)
    font.setKerning(True)
    font.setStyleStrategy(QtGui.QFont.PreferDefault)
    self.label_True_CM2.setFont(font)
    self.label_True_CM2.setObjectName("label_True_CM2")
    self.horizontalLayout_23.addWidget(self.label_True_CM2)
    self.tableWidget_CM2 = QtWidgets.QTableWidget(self.groupBox_confusionMatrixPlot)
    sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
    sizePolicy.setHorizontalStretch(0)
    sizePolicy.setVerticalStretch(0)
    sizePolicy.setHeightForWidth(self.tableWidget_CM2.sizePolicy().hasHeightForWidth())
    self.tableWidget_CM2.setSizePolicy(sizePolicy)
    self.tableWidget_CM2.setBaseSize(QtCore.QSize(250, 250))
    self.tableWidget_CM2.setObjectName("tableWidget_CM2")
    self.tableWidget_CM2.setColumnCount(0)
    self.tableWidget_CM2.setRowCount(0)
    self.horizontalLayout_23.addWidget(self.tableWidget_CM2)
    self.verticalLayout_12.addLayout(self.horizontalLayout_23)
    self.horizontalLayout_22 = QtWidgets.QHBoxLayout()
    self.horizontalLayout_22.setObjectName("horizontalLayout_22")
    self.pushButton_CM2_to_Clipboard = QtWidgets.QPushButton(self.groupBox_confusionMatrixPlot)
    self.pushButton_CM2_to_Clipboard.setObjectName("pushButton_CM2_to_Clipboard")
    self.horizontalLayout_22.addWidget(self.pushButton_CM2_to_Clipboard)
    self.label_Pred_CM2 = QtWidgets.QLabel(self.groupBox_confusionMatrixPlot)
    font = QtGui.QFont()
    font.setPointSize(8)
    font.setBold(True)
    font.setWeight(75)
    self.label_Pred_CM2.setFont(font)
    self.label_Pred_CM2.setFrameShape(QtWidgets.QFrame.NoFrame)
    self.label_Pred_CM2.setLineWidth(1)
    self.label_Pred_CM2.setMidLineWidth(0)
    self.label_Pred_CM2.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
    self.label_Pred_CM2.setIndent(-1)
    self.label_Pred_CM2.setObjectName("label_Pred_CM2")
    self.horizontalLayout_22.addWidget(self.label_Pred_CM2)
    self.verticalLayout_12.addLayout(self.horizontalLayout_22)
    self.horizontalLayout_24.addLayout(self.verticalLayout_12)
    self.gridLayout_16.addLayout(self.horizontalLayout_24, 0, 0, 1, 1)
    self.tableWidget_AccPrecSpec = QtWidgets.QTableWidget(self.groupBox_confusionMatrixPlot)
    sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
    sizePolicy.setHorizontalStretch(0)
    sizePolicy.setVerticalStretch(0)
    sizePolicy.setHeightForWidth(self.tableWidget_AccPrecSpec.sizePolicy().hasHeightForWidth())
    self.tableWidget_AccPrecSpec.setSizePolicy(sizePolicy)
    self.tableWidget_AccPrecSpec.setObjectName("tableWidget_AccPrecSpec")
    self.gridLayout_16.addWidget(self.tableWidget_AccPrecSpec, 1, 0, 1, 1)
    self.groupBox_probHistPlot = QtWidgets.QGroupBox(self.splitter_8)
    self.groupBox_probHistPlot.setObjectName("groupBox_probHistPlot")
    self.gridLayout_19 = QtWidgets.QGridLayout(self.groupBox_probHistPlot)
    self.gridLayout_19.setObjectName("gridLayout_19")
    self.widget_probHistPlot = pg.GraphicsLayoutWidget(self.groupBox_probHistPlot)#QtWidgets.QWidget(self.groupBox_probHistPlot)
    self.widget_probHistPlot.setObjectName("widget_probHistPlot")
    self.gridLayout_19.addWidget(self.widget_probHistPlot, 0, 0, 1, 1)
    self.groupBox_3rdPlot = QtWidgets.QGroupBox(self.splitter_8)
    self.groupBox_3rdPlot.setObjectName("groupBox_3rdPlot")
    self.gridLayout_18 = QtWidgets.QGridLayout(self.groupBox_3rdPlot)
    self.gridLayout_18.setObjectName("gridLayout_18")
    self.widget_3rdPlot = pg.GraphicsLayoutWidget(self.groupBox_3rdPlot)
    self.widget_3rdPlot.setObjectName("widget_3rdPlot")
    self.gridLayout_18.addWidget(self.widget_3rdPlot, 1, 0, 1, 1)
    self.gridLayout_23.addWidget(self.splitter_7, 0, 0, 1, 1)
    self.tabWidget_Modelbuilder.addTab(self.tab_AssessModel, "")

    ##################Tab Plotting and Peakdetermination##################

    self.tab_Plotting = QtWidgets.QWidget()
    self.tab_Plotting.setObjectName("tab_Plotting")
    self.gridLayout_25 = QtWidgets.QGridLayout(self.tab_Plotting)
    self.gridLayout_25.setObjectName("gridLayout_25")
    self.comboBox_chooseRtdcFile = QtWidgets.QComboBox(self.tab_Plotting)
    self.comboBox_chooseRtdcFile.setObjectName("comboBox_chooseRtdcFile")
    self.gridLayout_25.addWidget(self.comboBox_chooseRtdcFile, 0, 0, 1, 1)
    self.splitter_15 = QtWidgets.QSplitter(self.tab_Plotting)
    self.splitter_15.setOrientation(QtCore.Qt.Horizontal)
    self.splitter_15.setObjectName("splitter_15")
    self.splitter_14 = QtWidgets.QSplitter(self.splitter_15)
    self.splitter_14.setOrientation(QtCore.Qt.Vertical)
    self.splitter_14.setObjectName("splitter_14")
    self.groupBox_plottingregion = QtWidgets.QGroupBox(self.splitter_14)
    self.groupBox_plottingregion.setObjectName("groupBox_plottingregion")
    self.gridLayout_27 = QtWidgets.QGridLayout(self.groupBox_plottingregion)
    self.gridLayout_27.setObjectName("gridLayout_27")
    self.horizontalLayout_27 = QtWidgets.QHBoxLayout()
    self.horizontalLayout_27.setObjectName("horizontalLayout_27")
    self.comboBox_featurey = QtWidgets.QComboBox(self.groupBox_plottingregion)
    self.comboBox_featurey.setLayoutDirection(QtCore.Qt.LeftToRight)
    self.comboBox_featurey.setCurrentText("")
    self.comboBox_featurey.setObjectName("comboBox_featurey")
    self.horizontalLayout_27.addWidget(self.comboBox_featurey)
    self.verticalLayout_15 = QtWidgets.QVBoxLayout()
    self.verticalLayout_15.setObjectName("verticalLayout_15")
    self.splitter_13 = QtWidgets.QSplitter(self.groupBox_plottingregion)
    self.splitter_13.setOrientation(QtCore.Qt.Vertical)
    self.splitter_13.setObjectName("splitter_13")
    self.splitter_12 = QtWidgets.QSplitter(self.splitter_13)
    self.splitter_12.setOrientation(QtCore.Qt.Horizontal)
    self.splitter_12.setObjectName("splitter_12")
    self.widget_histx = pg.GraphicsLayoutWidget(self.splitter_12)
    self.widget_histx.resize(QtCore.QSize(251, 75))
    self.widget_histx.setObjectName("widget_histx")
    self.widget_infoBox = QtWidgets.QWidget(self.splitter_12)
    self.widget_infoBox.setObjectName("widget_infoBox")
    self.gridLayout_30 = QtWidgets.QGridLayout(self.widget_infoBox)
    self.gridLayout_30.setContentsMargins(0, 0, 0, 0)
    self.gridLayout_30.setObjectName("gridLayout_30")
    self.spinBox_cellInd = QtWidgets.QSpinBox(self.widget_infoBox)
    self.spinBox_cellInd.setMaximumSize(QtCore.QSize(16777215, 22))
    self.spinBox_cellInd.setMaximum(999999999)
    self.spinBox_cellInd.setObjectName("spinBox_cellInd")
    self.gridLayout_30.addWidget(self.spinBox_cellInd, 1, 0, 1, 1)
    self.horizontalSlider_cellInd = QtWidgets.QSlider(self.widget_infoBox)
    self.horizontalSlider_cellInd.setOrientation(QtCore.Qt.Horizontal)
    self.horizontalSlider_cellInd.setObjectName("horizontalSlider_cellInd")
    self.gridLayout_30.addWidget(self.horizontalSlider_cellInd, 0, 0, 1, 1)
    self.splitter_9 = QtWidgets.QSplitter(self.splitter_13)
    self.splitter_9.setOrientation(QtCore.Qt.Horizontal)
    self.splitter_9.setObjectName("splitter_9")
    self.widget_scatter = pg.GraphicsLayoutWidget(self.splitter_9)
    sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
    sizePolicy.setHorizontalStretch(1)
    sizePolicy.setVerticalStretch(1)
    sizePolicy.setHeightForWidth(self.widget_scatter.sizePolicy().hasHeightForWidth())
    self.widget_scatter.setSizePolicy(sizePolicy)
    #self.widget_scatter.setMinimumSize(QtCore.QSize(251, 251))
    self.widget_scatter.setSizeIncrement(QtCore.QSize(1, 1))
    self.widget_scatter.setBaseSize(QtCore.QSize(1, 1))
    self.widget_scatter.setObjectName("widget_scatter")
    self.widget_histy = pg.GraphicsLayoutWidget(self.splitter_9)
    #self.widget_histy.setMinimumSize(QtCore.QSize(75, 251))
    self.widget_histy.setObjectName("widget_histy")
    self.verticalLayout_15.addWidget(self.splitter_13)
    self.comboBox_featurex = QtWidgets.QComboBox(self.groupBox_plottingregion)
    self.comboBox_featurex.setMinimumSize(QtCore.QSize(326, 30))
    self.comboBox_featurex.setObjectName("comboBox_featurex")
    self.verticalLayout_15.addWidget(self.comboBox_featurex)
    self.horizontalLayout_27.addLayout(self.verticalLayout_15)
    self.gridLayout_27.addLayout(self.horizontalLayout_27, 0, 0, 1, 1)
    self.groupBox_plottingOptions = QtWidgets.QGroupBox(self.splitter_14)
    self.groupBox_plottingOptions.setObjectName("groupBox_plottingOptions")
    self.gridLayout_26 = QtWidgets.QGridLayout(self.groupBox_plottingOptions)
    self.gridLayout_26.setObjectName("gridLayout_26")
    self.verticalLayout_14 = QtWidgets.QVBoxLayout()
    self.verticalLayout_14.setObjectName("verticalLayout_14")
    self.horizontalLayout_29 = QtWidgets.QHBoxLayout()
    self.horizontalLayout_29.setObjectName("horizontalLayout_29")
    self.checkBox_fl1 = QtWidgets.QCheckBox(self.groupBox_plottingOptions)
    self.checkBox_fl1.setObjectName("checkBox_fl1")
    self.horizontalLayout_29.addWidget(self.checkBox_fl1)
    self.checkBox_fl2 = QtWidgets.QCheckBox(self.groupBox_plottingOptions)
    self.checkBox_fl2.setObjectName("checkBox_fl2")
    self.horizontalLayout_29.addWidget(self.checkBox_fl2)
    self.checkBox_fl3 = QtWidgets.QCheckBox(self.groupBox_plottingOptions)
    self.checkBox_fl3.setObjectName("checkBox_fl3")
    self.horizontalLayout_29.addWidget(self.checkBox_fl3)
    self.checkBox_centroid = QtWidgets.QCheckBox(self.groupBox_plottingOptions)
    self.checkBox_centroid.setObjectName("checkBox_centroid")
    self.horizontalLayout_29.addWidget(self.checkBox_centroid)

    self.verticalLayout_14.addLayout(self.horizontalLayout_29)
    self.horizontalLayout_26 = QtWidgets.QHBoxLayout()
    self.horizontalLayout_26.setObjectName("horizontalLayout_26")
    self.label_coloring = QtWidgets.QLabel(self.groupBox_plottingOptions)
    self.label_coloring.setObjectName("label_coloring")
    self.horizontalLayout_26.addWidget(self.label_coloring)
    self.comboBox_coloring = QtWidgets.QComboBox(self.groupBox_plottingOptions)
    self.comboBox_coloring.setObjectName("comboBox_coloring")
    self.horizontalLayout_26.addWidget(self.comboBox_coloring)
    self.checkBox_colorLog = QtWidgets.QCheckBox(self.groupBox_plottingOptions)
    self.checkBox_colorLog.setObjectName("checkBox_colorLog")
    self.horizontalLayout_26.addWidget(self.checkBox_colorLog)
    self.pushButton_updateScatterPlot = QtWidgets.QPushButton(self.groupBox_plottingOptions)
    self.pushButton_updateScatterPlot.setObjectName("pushButton_updateScatterPlot")
    self.horizontalLayout_26.addWidget(self.pushButton_updateScatterPlot)
    self.verticalLayout_14.addLayout(self.horizontalLayout_26)
    self.gridLayout_26.addLayout(self.verticalLayout_14, 0, 0, 1, 1)
    self.groupBox = QtWidgets.QGroupBox(self.splitter_14)
    self.groupBox.setObjectName("groupBox")
    self.gridLayout_33 = QtWidgets.QGridLayout(self.groupBox)
    self.gridLayout_33.setObjectName("gridLayout_33")
    self.textBrowser_fileInfo = QtWidgets.QTextBrowser(self.groupBox)
    self.textBrowser_fileInfo.setObjectName("textBrowser_fileInfo")
    self.gridLayout_33.addWidget(self.textBrowser_fileInfo, 0, 0, 1, 1)
    self.tabWidget_filter_peakdet = QtWidgets.QTabWidget(self.splitter_15)
    self.tabWidget_filter_peakdet.setObjectName("tabWidget_filter_peakdet")
    self.tab_filter = QtWidgets.QWidget()
    self.tab_filter.setObjectName("tab_filter")
    self.gridLayout_24 = QtWidgets.QGridLayout(self.tab_filter)
    self.gridLayout_24.setObjectName("gridLayout_24")
    self.tableWidget_filterOptions = QtWidgets.QTableWidget(self.tab_filter)
    self.tableWidget_filterOptions.setObjectName("tableWidget_filterOptions")
    self.tableWidget_filterOptions.setColumnCount(0)
    self.tableWidget_filterOptions.setRowCount(0)
    self.gridLayout_24.addWidget(self.tableWidget_filterOptions, 0, 0, 1, 1)
    self.tabWidget_filter_peakdet.addTab(self.tab_filter, "")
    self.tab_peakdet = QtWidgets.QWidget()
    self.tab_peakdet.setObjectName("tab_peakdet")
    self.gridLayout_29 = QtWidgets.QGridLayout(self.tab_peakdet)
    self.gridLayout_29.setObjectName("gridLayout_29")
    self.splitter_10 = QtWidgets.QSplitter(self.tab_peakdet)
    self.splitter_10.setOrientation(QtCore.Qt.Vertical)
    self.splitter_10.setObjectName("splitter_10")
    self.groupBox_showCell = QtWidgets.QGroupBox(self.splitter_10)
    self.groupBox_showCell.setObjectName("groupBox_showCell")
    self.gridLayout_32 = QtWidgets.QGridLayout(self.groupBox_showCell)
    self.gridLayout_32.setObjectName("gridLayout_32")
    self.widget_showFltrace = pg.GraphicsLayoutWidget(self.groupBox_showCell)
    self.widget_showFltrace.setMinimumSize(QtCore.QSize(0, 81))
    #self.widget_showFltrace.setMaximumSize(QtCore.QSize(16777215, 81))
    self.widget_showFltrace.setObjectName("widget_showFltrace")
    self.gridLayout_32.addWidget(self.widget_showFltrace, 1, 0, 1, 1)
    self.widget_showCell = pg.ImageView(self.groupBox_showCell)
    self.widget_showCell.setMinimumSize(QtCore.QSize(0, 91))
    #self.widget_showCell.setMaximumSize(QtCore.QSize(16777215, 91))
    self.widget_showCell.ui.histogram.hide()
    self.widget_showCell.ui.roiBtn.hide()
    self.widget_showCell.ui.menuBtn.hide()
    self.widget_showCell.setObjectName("widget_showCell")
    self.gridLayout_32.addWidget(self.widget_showCell, 0, 0, 1, 1)
    
    self.groupBox_showSelectedPeaks = QtWidgets.QGroupBox(self.splitter_10)
    self.groupBox_showSelectedPeaks.setObjectName("groupBox_showSelectedPeaks")
    self.gridLayout_28 = QtWidgets.QGridLayout(self.groupBox_showSelectedPeaks)
    self.gridLayout_28.setObjectName("gridLayout_28")
    self.splitter_11 = QtWidgets.QSplitter(self.groupBox_showSelectedPeaks)
    self.splitter_11.setOrientation(QtCore.Qt.Horizontal)
    self.splitter_11.setObjectName("splitter_11")
    self.widget = QtWidgets.QWidget(self.splitter_11)
    self.widget.setObjectName("widget")
    self.verticalLayout_18 = QtWidgets.QVBoxLayout(self.widget)
    self.verticalLayout_18.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
    self.verticalLayout_18.setContentsMargins(0, 0, 0, 0)
    self.verticalLayout_18.setObjectName("verticalLayout_18")
    self.horizontalLayout_30 = QtWidgets.QHBoxLayout()
    self.horizontalLayout_30.setObjectName("horizontalLayout_30")
    self.pushButton_selectPeakPos = QtWidgets.QPushButton(self.widget)
    self.pushButton_selectPeakPos.setMinimumSize(QtCore.QSize(50, 28))
    self.pushButton_selectPeakPos.setMaximumSize(QtCore.QSize(50, 28))
    self.pushButton_selectPeakPos.setObjectName("pushButton_selectPeakPos")
    self.horizontalLayout_30.addWidget(self.pushButton_selectPeakPos)
    self.pushButton_selectPeakRange = QtWidgets.QPushButton(self.widget)
    self.pushButton_selectPeakRange.setMinimumSize(QtCore.QSize(50, 28))
    self.pushButton_selectPeakRange.setMaximumSize(QtCore.QSize(50, 28))
    self.pushButton_selectPeakRange.setObjectName("pushButton_selectPeakRange")
    self.horizontalLayout_30.addWidget(self.pushButton_selectPeakRange)
    self.verticalLayout_18.addLayout(self.horizontalLayout_30)
    self.label_automatic = QtWidgets.QLabel(self.widget)
    self.label_automatic.setObjectName("label_automatic")
    self.verticalLayout_18.addWidget(self.label_automatic)
    self.horizontalLayout_28 = QtWidgets.QHBoxLayout()
    self.horizontalLayout_28.setObjectName("horizontalLayout_28")
    self.pushButton_highestXPercent = QtWidgets.QPushButton(self.widget)
    self.pushButton_highestXPercent.setMinimumSize(QtCore.QSize(82, 28))
    self.pushButton_highestXPercent.setMaximumSize(QtCore.QSize(82, 28))
    self.pushButton_highestXPercent.setObjectName("pushButton_highestXPercent")
    self.horizontalLayout_28.addWidget(self.pushButton_highestXPercent)
    self.doubleSpinBox_highestXPercent = QtWidgets.QDoubleSpinBox(self.widget)
    self.doubleSpinBox_highestXPercent.setMinimumSize(QtCore.QSize(51, 22))
    self.doubleSpinBox_highestXPercent.setMaximumSize(QtCore.QSize(51, 22))
    self.doubleSpinBox_highestXPercent.setObjectName("doubleSpinBox_highestXPercent")
    self.horizontalLayout_28.addWidget(self.doubleSpinBox_highestXPercent)
    self.verticalLayout_18.addLayout(self.horizontalLayout_28)
    spacerItem = QtWidgets.QSpacerItem(157, 13, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
    self.verticalLayout_18.addItem(spacerItem)
    self.label_remove = QtWidgets.QLabel(self.widget)
    self.label_remove.setObjectName("label_remove")
    self.verticalLayout_18.addWidget(self.label_remove)
    self.horizontalLayout_31 = QtWidgets.QHBoxLayout()
    self.horizontalLayout_31.setObjectName("horizontalLayout_31")
    self.pushButton_removeSelectedPeaks = QtWidgets.QPushButton(self.widget)
    self.pushButton_removeSelectedPeaks.setMinimumSize(QtCore.QSize(60, 28))
    self.pushButton_removeSelectedPeaks.setMaximumSize(QtCore.QSize(60, 28))
    self.pushButton_removeSelectedPeaks.setObjectName("pushButton_removeSelectedPeaks")
    self.horizontalLayout_31.addWidget(self.pushButton_removeSelectedPeaks)
    self.pushButton_removeAllPeaks = QtWidgets.QPushButton(self.widget)
    self.pushButton_removeAllPeaks.setMinimumSize(QtCore.QSize(40, 28))
    self.pushButton_removeAllPeaks.setMaximumSize(QtCore.QSize(40, 28))
    self.pushButton_removeAllPeaks.setObjectName("pushButton_removeAllPeaks")
    self.horizontalLayout_31.addWidget(self.pushButton_removeAllPeaks)
    self.verticalLayout_18.addLayout(self.horizontalLayout_31)
    self.widget_showSelectedPeaks = pg.GraphicsLayoutWidget(self.splitter_11)
    self.widget_showSelectedPeaks.setObjectName("widget_showSelectedPeaks")
    self.tableWidget_showSelectedPeaks = QtWidgets.QTableWidget(self.splitter_11)
    self.tableWidget_showSelectedPeaks.setObjectName("tableWidget_showSelectedPeaks")
    self.gridLayout_28.addWidget(self.splitter_11, 0, 0, 1, 1)


    self.groupBox_peakDetModel = QtWidgets.QGroupBox(self.splitter_10)
    self.groupBox_peakDetModel.setObjectName("groupBox_peakDetModel")
    self.gridLayout_31 = QtWidgets.QGridLayout(self.groupBox_peakDetModel)
    self.gridLayout_31.setObjectName("gridLayout_31")
    self.horizontalLayout_25 = QtWidgets.QHBoxLayout()
    self.horizontalLayout_25.setObjectName("horizontalLayout_25")
    self.verticalLayout_16 = QtWidgets.QVBoxLayout()
    self.verticalLayout_16.setObjectName("verticalLayout_16")
    self.comboBox_peakDetModel = QtWidgets.QComboBox(self.groupBox_peakDetModel)
    self.comboBox_peakDetModel.setObjectName("comboBox_peakDetModel")
    self.verticalLayout_16.addWidget(self.comboBox_peakDetModel)
    self.pushButton_fitPeakDetModel = QtWidgets.QPushButton(self.groupBox_peakDetModel)
    self.pushButton_fitPeakDetModel.setObjectName("pushButton_fitPeakDetModel")
    self.verticalLayout_16.addWidget(self.pushButton_fitPeakDetModel)
    self.pushButton_SavePeakDetModel = QtWidgets.QPushButton(self.groupBox_peakDetModel)
    self.pushButton_SavePeakDetModel.setObjectName("pushButton_SavePeakDetModel")
    self.verticalLayout_16.addWidget(self.pushButton_SavePeakDetModel)
    self.pushButton_loadPeakDetModel = QtWidgets.QPushButton(self.groupBox_peakDetModel)
    self.pushButton_loadPeakDetModel.setObjectName("pushButton_loadPeakDetModel")
    self.verticalLayout_16.addWidget(self.pushButton_loadPeakDetModel)
    self.horizontalLayout_25.addLayout(self.verticalLayout_16)
    self.tableWidget_peakModelParameters = QtWidgets.QTableWidget(self.groupBox_peakDetModel)
    self.tableWidget_peakModelParameters.setObjectName("tableWidget_peakModelParameters")
    self.tableWidget_peakModelParameters.setColumnCount(0)
    self.tableWidget_peakModelParameters.setRowCount(0)
    self.horizontalLayout_25.addWidget(self.tableWidget_peakModelParameters)
    self.gridLayout_31.addLayout(self.horizontalLayout_25, 0, 0, 1, 1)
    self.verticalLayout_17 = QtWidgets.QVBoxLayout()
    self.verticalLayout_17.setObjectName("verticalLayout_17")
    self.radioButton_exportSelected = QtWidgets.QRadioButton(self.groupBox_peakDetModel)
    self.radioButton_exportSelected.setChecked(True)
    self.radioButton_exportSelected.setObjectName("radioButton_exportSelected")
    self.verticalLayout_17.addWidget(self.radioButton_exportSelected)
    self.radioButton_exportAll = QtWidgets.QRadioButton(self.groupBox_peakDetModel)
    self.radioButton_exportAll.setObjectName("radioButton_exportAll")
    self.verticalLayout_17.addWidget(self.radioButton_exportAll)
    self.comboBox_toFlOrUserdef = QtWidgets.QComboBox(self.groupBox_peakDetModel)
    self.comboBox_toFlOrUserdef.setObjectName("comboBox_toFlOrUserdef")
    self.verticalLayout_17.addWidget(self.comboBox_toFlOrUserdef)
    self.pushButton_export = QtWidgets.QPushButton(self.groupBox_peakDetModel)
    self.pushButton_export.setObjectName("pushButton_export")
    self.verticalLayout_17.addWidget(self.pushButton_export)
    self.gridLayout_31.addLayout(self.verticalLayout_17, 0, 1, 1, 1)
    self.gridLayout_29.addWidget(self.splitter_10, 0, 0, 1, 1)
    self.tabWidget_filter_peakdet.addTab(self.tab_peakdet, "")
    self.tab_defineModel = QtWidgets.QWidget()
    self.tab_defineModel.setObjectName("tab_defineModel")
    self.tabWidget_filter_peakdet.addTab(self.tab_defineModel, "")
    self.gridLayout_25.addWidget(self.splitter_15, 1, 0, 1, 1)
    self.tabWidget_Modelbuilder.addTab(self.tab_Plotting, "")




    #####################Icons##################
    icon = QtGui.QIcon()
    icon.addPixmap(QtGui.QPixmap(os.path.join(dir_root,"art",Default_dict["Icon theme"],"thumb.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
    self.pushButton_AssessModel.setIcon(icon)





    ####################Tab Python Editor/Console##########################
    self.tab_python = QtWidgets.QWidget()
    self.tab_python.setObjectName("tab_python")
    self.gridLayout_41 = QtWidgets.QGridLayout(self.tab_python)
    self.gridLayout_41.setObjectName("gridLayout_41")
    self.verticalLayout_python_1 = QtWidgets.QVBoxLayout()
    self.verticalLayout_python_1.setObjectName("verticalLayout_python_1")
    self.groupBox_pythonMenu = QtWidgets.QGroupBox(self.tab_python)
    self.groupBox_pythonMenu.setMaximumSize(QtCore.QSize(16777215, 71))
    self.groupBox_pythonMenu.setObjectName("groupBox_pythonMenu")
    self.gridLayout_40 = QtWidgets.QGridLayout(self.groupBox_pythonMenu)
    self.gridLayout_40.setObjectName("gridLayout_40")
    self.horizontalLayout_pythonMenu = QtWidgets.QHBoxLayout()
    self.horizontalLayout_pythonMenu.setObjectName("horizontalLayout_pythonMenu")
    self.label_pythonCurrentFile = QtWidgets.QLabel(self.groupBox_pythonMenu)
    self.label_pythonCurrentFile.setObjectName("label_pythonCurrentFile")
    self.horizontalLayout_pythonMenu.addWidget(self.label_pythonCurrentFile)
    self.lineEdit_pythonCurrentFile = QtWidgets.QLineEdit(self.groupBox_pythonMenu)
    self.lineEdit_pythonCurrentFile.setEnabled(False)
    self.lineEdit_pythonCurrentFile.setObjectName("lineEdit_pythonCurrentFile")
    self.horizontalLayout_pythonMenu.addWidget(self.lineEdit_pythonCurrentFile)
    self.gridLayout_40.addLayout(self.horizontalLayout_pythonMenu, 0, 0, 1, 1)
    self.verticalLayout_python_1.addWidget(self.groupBox_pythonMenu)
    self.splitter_python_1 = QtWidgets.QSplitter(self.tab_python)
    self.splitter_python_1.setOrientation(QtCore.Qt.Horizontal)
    self.splitter_python_1.setObjectName("splitter_python_1")
    self.groupBox_pythonEditor = QtWidgets.QGroupBox(self.splitter_python_1)
    self.groupBox_pythonEditor.setObjectName("groupBox_pythonEditor")
    self.gridLayout_38 = QtWidgets.QGridLayout(self.groupBox_pythonEditor)
    self.gridLayout_38.setObjectName("gridLayout_38")
    self.verticalLayout_editor_1 = QtWidgets.QVBoxLayout()
    self.verticalLayout_editor_1.setObjectName("verticalLayout_editor_1")
    self.horizontalLayout_editor_1 = QtWidgets.QHBoxLayout()
    self.horizontalLayout_editor_1.setObjectName("horizontalLayout_editor_1")
    self.pushButton_pythonInOpen = QtWidgets.QPushButton(self.groupBox_pythonEditor)
    self.pushButton_pythonInOpen.setObjectName("pushButton_pythonInOpen")
    self.horizontalLayout_editor_1.addWidget(self.pushButton_pythonInOpen)
    self.pushButton_pythonSaveAs = QtWidgets.QPushButton(self.groupBox_pythonEditor)
    self.pushButton_pythonSaveAs.setObjectName("pushButton_pythonSaveAs")
    self.horizontalLayout_editor_1.addWidget(self.pushButton_pythonSaveAs)
    self.pushButton_pythonInClear = QtWidgets.QPushButton(self.groupBox_pythonEditor)
    self.pushButton_pythonInClear.setObjectName("pushButton_pythonInClear")
    self.horizontalLayout_editor_1.addWidget(self.pushButton_pythonInClear)
    self.pushButton_pythonInRun = QtWidgets.QPushButton(self.groupBox_pythonEditor)
    self.pushButton_pythonInRun.setObjectName("pushButton_pythonInRun")
    self.horizontalLayout_editor_1.addWidget(self.pushButton_pythonInRun)
    self.verticalLayout_editor_1.addLayout(self.horizontalLayout_editor_1)
    self.plainTextEdit_pythonIn = QtWidgets.QPlainTextEdit(self.groupBox_pythonEditor)
    self.plainTextEdit_pythonIn.setObjectName("plainTextEdit_pythonIn")
    self.verticalLayout_editor_1.addWidget(self.plainTextEdit_pythonIn)
    self.gridLayout_38.addLayout(self.verticalLayout_editor_1, 0, 0, 1, 1)
    self.groupBox_pythonConsole = QtWidgets.QGroupBox(self.splitter_python_1)
    self.groupBox_pythonConsole.setObjectName("groupBox_pythonConsole")
    self.gridLayout_39 = QtWidgets.QGridLayout(self.groupBox_pythonConsole)
    self.gridLayout_39.setObjectName("gridLayout_39")
    self.verticalLayout_console_1 = QtWidgets.QVBoxLayout()
    self.verticalLayout_console_1.setObjectName("verticalLayout_console_1")
    self.horizontalLayout_console_1 = QtWidgets.QHBoxLayout()
    self.horizontalLayout_console_1.setObjectName("horizontalLayout_console_1")
    self.pushButton_pythonOutClear = QtWidgets.QPushButton(self.groupBox_pythonConsole)
    self.pushButton_pythonOutClear.setObjectName("pushButton_pythonOutClear")
    self.horizontalLayout_console_1.addWidget(self.pushButton_pythonOutClear)
    self.pushButton_pythonOutRun = QtWidgets.QPushButton(self.groupBox_pythonConsole)
    self.pushButton_pythonOutRun.setObjectName("pushButton_pythonOutRun")
    self.horizontalLayout_console_1.addWidget(self.pushButton_pythonOutRun)
    self.verticalLayout_console_1.addLayout(self.horizontalLayout_console_1)
    self.textBrowser_pythonOut = QtWidgets.QTextBrowser(self.groupBox_pythonConsole)
    self.textBrowser_pythonOut.setEnabled(True)
    self.textBrowser_pythonOut.setObjectName("textBrowser_pythonOut")
    self.verticalLayout_console_1.addWidget(self.textBrowser_pythonOut)
    self.gridLayout_39.addLayout(self.verticalLayout_console_1, 0, 0, 1, 1)
    self.verticalLayout_python_1.addWidget(self.splitter_python_1)
    self.gridLayout_41.addLayout(self.verticalLayout_python_1, 0, 0, 1, 1)
    self.tabWidget_Modelbuilder.addTab(self.tab_python, "")
    

    
    
    
    
    
    
    
    
    




    ######################Connections######################################
    self.doubleSpinBox_learningRate.setEnabled(False)
    self.spinBox_trainLastNOnly.setEnabled(False)
    self.lineEdit_dropout.setEnabled(False)
    self.groupBox_learningRate.toggled['bool'].connect(self.doubleSpinBox_learningRate.setEnabled)        
    self.checkBox_trainLastNOnly.toggled['bool'].connect(self.spinBox_trainLastNOnly.setEnabled)
    self.checkBox_dropout.toggled['bool'].connect(self.lineEdit_dropout.setEnabled)













    #######################################################################
    ##########################Manual manipulation##########################
    ##########################Manual manipulation##########################
    #######################################################################
    ###########this makes it easier if the Ui should be changed############



    ####################Define Model####################################
    #self.label_Crop.setChecked(True)
    #self.label_Crop.stateChanged.connect(self.activate_deactivate_spinbox)
    
    self.spinBox_imagecrop.valueChanged.connect(self.delete_ram)
    self.colorModes = ["Grayscale","RGB"]
    self.comboBox_GrayOrRGB.addItems(self.colorModes)
    self.comboBox_GrayOrRGB.setCurrentIndex(0)             
    self.comboBox_GrayOrRGB.currentIndexChanged.connect(self.gray_or_rgb_augmentation)            
    #By default, Color Mode is Grayscale. Therefore switch off saturation and hue
    self.checkBox_contrast.setEnabled(True)
    self.checkBox_contrast.setChecked(True)
    self.doubleSpinBox_contrastLower.setEnabled(True)
    self.doubleSpinBox_contrastHigher.setEnabled(True)
    self.checkBox_saturation.setEnabled(False)
    self.checkBox_saturation.setChecked(False)
    self.doubleSpinBox_saturationLower.setEnabled(False)
    self.doubleSpinBox_saturationHigher.setEnabled(False)
    self.checkBox_hue.setEnabled(False)
    self.checkBox_hue.setChecked(False)
    self.doubleSpinBox_hueDelta.setEnabled(False)


    ###########################Expert mode Tab################################
    self.spinBox_batchSize.setMinimum(1)       
    self.spinBox_batchSize.setMaximum(1E6)       
    self.spinBox_batchSize.setValue(Default_dict["spinBox_batchSize"])       
    self.spinBox_epochs.setMinimum(1)       
    self.spinBox_epochs.setMaximum(1E6)       
    self.spinBox_epochs.setValue(1)       
    self.doubleSpinBox_learningRate.setDecimals(9)
    self.doubleSpinBox_learningRate.setMinimum(0.0)       
    self.doubleSpinBox_learningRate.setMaximum(1E6)       
    self.doubleSpinBox_learningRate.setValue(0.001)       
    self.doubleSpinBox_learningRate.setSingleStep(0.0001)
    self.spinBox_trainLastNOnly.setMinimum(0)       
    self.spinBox_trainLastNOnly.setMaximum(1E6)       
    self.spinBox_trainLastNOnly.setValue(0)    
    self.checkBox_trainDenseOnly.setChecked(False)
    self.checkBox_partialTrainability.toggled.connect(self.partialtrainability_activated)
    self.checkBox_lossW.clicked.connect(lambda on_or_off: self.lossWeights_activated(on_or_off,-1))
    self.pushButton_lossW.clicked.connect(lambda: self.lossWeights_popup(-1))
    self.pushButton_optimizer.clicked.connect(lambda: self.optimizer_change_settings_popup(-1))


    ###########################History Tab################################
    self.tableWidget_HistoryItems.doubleClicked.connect(self.tableWidget_HistoryItems_dclick)
    conversion_methods_source = ["Keras TensorFlow", "Frozen TensorFlow .pb"]
    conversion_methods_target = [".nnet","Frozen TensorFlow .pb", "Optimized TensorFlow .pb", "ONNX (via keras2onnx)", "ONNX (via MMdnn)","CoreML", "PyTorch Script","Caffe Script","CNTK Script","MXNet Script","ONNX Script","TensorFlow Script","Keras Script"]
    self.comboBox_convertTo.addItems(conversion_methods_target)
    self.comboBox_convertTo.setMinimumSize(QtCore.QSize(200,22))
    self.comboBox_convertTo.setMaximumSize(QtCore.QSize(200, 22))
    width=self.comboBox_convertTo.fontMetrics().boundingRect(max(conversion_methods_target, key=len)).width()
    self.comboBox_convertTo.view().setFixedWidth(width+10)
    self.combobox_initial_format.setCurrentIndex(0)             
    #self.comboBox_convertTo.setEnabled(False)
    
    self.combobox_initial_format.addItems(conversion_methods_source)
    self.combobox_initial_format.setMinimumSize(QtCore.QSize(200,22))
    self.combobox_initial_format.setMaximumSize(QtCore.QSize(200, 22))
    width=self.combobox_initial_format.fontMetrics().boundingRect(max(conversion_methods_source, key=len)).width()
    self.combobox_initial_format.view().setFixedWidth(width+10)             
    self.combobox_initial_format.setCurrentIndex(0)             
    #self.combobox_initial_format.setEnabled(False)


    ###########################Assess Model################################
    self.comboBox_loadedRGBorGray.addItems(["Grayscale","RGB"])
    self.groupBox_loadModel.setMaximumSize(QtCore.QSize(16777215, 120))
    self.label_ModelIndex_2.setMinimumSize(QtCore.QSize(68, 25))
    self.label_ModelIndex_2.setMaximumSize(QtCore.QSize(68, 25))
    self.spinBox_ModelIndex_2.setEnabled(False) 
    self.spinBox_ModelIndex_2.setMinimum(0)
    self.spinBox_ModelIndex_2.setMaximum(9E8)
    self.spinBox_ModelIndex_2.setMinimumSize(QtCore.QSize(40, 22))
    self.spinBox_ModelIndex_2.setMaximumSize(QtCore.QSize(75, 22))
    self.spinBox_Crop_2.setMinimumSize(QtCore.QSize(40, 22))
    self.spinBox_Crop_2.setMaximumSize(QtCore.QSize(50, 22))
    self.spinBox_Crop_2.setMinimum(1)
    self.spinBox_Crop_2.setMaximum(9E8)
    self.spinBox_OutClasses_2.setMinimumSize(QtCore.QSize(40, 22))
    self.spinBox_OutClasses_2.setMaximumSize(QtCore.QSize(50, 22))
    self.spinBox_OutClasses_2.setMinimum(1)
    self.spinBox_OutClasses_2.setMaximum(9E8)
    
    self.lineEdit_ModelSelection_2.setMinimumSize(QtCore.QSize(0, 20))
    self.lineEdit_ModelSelection_2.setMaximumSize(QtCore.QSize(16777215, 20))

    self.pushButton_LoadModel_2.setMinimumSize(QtCore.QSize(123, 24))
    self.pushButton_LoadModel_2.setMaximumSize(QtCore.QSize(123, 24))
    self.pushButton_LoadModel_2.clicked.connect(self.assessmodel_tab_load_model)

    self.norm_methods = Default_dict["norm_methods"] #["None","Div. by 255", "StdScaling using mean and std of each image individually","StdScaling using mean and std of all training data"]
    self.comboBox_Normalization_2.addItems(self.norm_methods)
    self.comboBox_Normalization_2.setMinimumSize(QtCore.QSize(200,22))
    self.comboBox_Normalization_2.setMaximumSize(QtCore.QSize(200, 22))
    width=self.comboBox_Normalization_2.fontMetrics().boundingRect(max(self.norm_methods, key=len)).width()
    self.comboBox_Normalization_2.view().setFixedWidth(width+10)             

    self.pushButton_ImportValidFromNpy.setMinimumSize(QtCore.QSize(65, 28))
    self.pushButton_ImportValidFromNpy.clicked.connect(self.import_valid_from_rtdc)
    self.pushButton_ExportValidToNpy.setMinimumSize(QtCore.QSize(55, 0))
    self.pushButton_ExportValidToNpy.clicked.connect(self.export_valid_to_rtdc)

    self.lineEdit_InferenceTime.setMinimumSize(QtCore.QSize(0, 30))
    self.lineEdit_InferenceTime.setMaximumSize(QtCore.QSize(16777215, 30))
    self.spinBox_inftime_nr_images.setMinimum(10)
    self.spinBox_inftime_nr_images.setMaximum(9E8)
    self.spinBox_inftime_nr_images.setValue(1000)
    
    self.groupBox_validData.setMaximumSize(QtCore.QSize(400, 250))
    self.tableWidget_Info_2.setColumnCount(0)
    self.tableWidget_Info_2.setRowCount(0)
    self.tableWidget_Info_2.clicked.connect(self.tableWidget_Info_2_click)
    self.groupBox_settings.setMaximumSize(QtCore.QSize(16777215, 250))
    self.spinBox_indexOfInterest.setMinimum(0)
    self.spinBox_indexOfInterest.setMaximum(9E8)

    self.doubleSpinBox_sortingThresh.setMinimum(0)
    self.doubleSpinBox_sortingThresh.setMaximum(1)
    self.doubleSpinBox_sortingThresh.setValue(0.5)
    self.doubleSpinBox_sortingThresh.setSingleStep(0.1)
    self.doubleSpinBox_sortingThresh.setDecimals(5)

    self.comboBox_selectData.setMinimumSize(QtCore.QSize(200,22))
    self.comboBox_selectData.setMaximumSize(QtCore.QSize(200, 32))

    
    #3rd Plot
    items_3rd_plot = ["None","Conc. vs. Threshold","Enrichment vs. Threshold","Yield vs. Threshold","ROC-AUC","Precision-Recall"]
    self.comboBox_3rdPlot.addItems(items_3rd_plot)
    self.comboBox_3rdPlot.setMinimumSize(QtCore.QSize(100,22))
    self.comboBox_3rdPlot.setMaximumSize(QtCore.QSize(100,22))
    width=self.comboBox_3rdPlot.fontMetrics().boundingRect(max(items_3rd_plot, key=len)).width()
    self.comboBox_3rdPlot.view().setFixedWidth(width+10)             
    self.comboBox_3rdPlot.currentTextChanged.connect(self.thirdplot)
    
    self.spinBox_Indx1.setMinimum(0)
    self.spinBox_Indx1.setMaximum(9E8)
    self.spinBox_Indx1.setEnabled(False)
    self.spinBox_Indx2.setMinimum(0)
    self.spinBox_Indx2.setMaximum(9E8)
    self.spinBox_Indx2.setEnabled(False)
#        self.groupBox_confusionMatrixPlot.setMinimumSize(QtCore.QSize(100, 16777215))
#        self.groupBox_confusionMatrixPlot.setMaximumSize(QtCore.QSize(600, 16777215))
    self.tableWidget_CM1.setColumnCount(0)
    self.tableWidget_CM1.setRowCount(0)
    self.tableWidget_CM1.doubleClicked.connect(self.cm_interaction)
    self.tableWidget_CM2.setColumnCount(0)
    self.tableWidget_CM2.setRowCount(0)
    self.tableWidget_CM2.doubleClicked.connect(self.cm_interaction)
    self.pushButton_CM1_to_Clipboard.clicked.connect(lambda: self.copy_cm_to_clipboard(1)) #1 tells the function to connect to CM1
    self.pushButton_CM2_to_Clipboard.clicked.connect(lambda: self.copy_cm_to_clipboard(2)) #2 tells the function to connect to CM2
    self.pushButton_CompInfTime.clicked.connect(self.inference_time)
    self.pushButton_AssessModel.clicked.connect(self.assess_model_plotting)
    #self.comboBox_probability_histogram.clicked.connect(self.probability_histogram)
    self.comboBox_probability_histogram.addItems(["Style1","Style2","Style3","Style4","Style5"])

    ##################Plot/Peak Tab########################################
    self.tabWidget_filter_peakdet.setCurrentIndex(1)
    self.comboBox_chooseRtdcFile.currentIndexChanged.connect(self.update_comboBox_feature_xy)
    self.pushButton_updateScatterPlot.clicked.connect(self.updateScatterPlot)        
    font = QtGui.QFont()
    font.setBold(True)
    font.setWeight(75)  
    self.pushButton_updateScatterPlot.setFont(font)        
    #set infobox to minimal size
    #self.widget_infoBox.resize(QtCore.QSize(100, 100))
    self.widget_infoBox.setMaximumSize(QtCore.QSize(100, 100))
    self.comboBox_featurey.setMinimumSize(QtCore.QSize(30, 326))
    self.comboBox_featurey.setMaximumSize(QtCore.QSize(75, 326))
    self.comboBox_featurey.view().setFixedWidth(150)#Wider box for the dropdown text

    #Add plot        
    self.scatter_xy = self.widget_scatter.addPlot()
    self.scatter_xy.showGrid(x=True,y=True)
   
    #Fill histogram for x-axis; widget_histx
    self.hist_x = self.widget_histx.addPlot()
    #hide the x-axis
    self.hist_x.hideAxis('bottom')
    self.hist_x.setXLink(self.scatter_xy) ## test linking by name
    self.hist_x.showGrid(x=True,y=True)

    #Initiate histogram for y-axis; widget_histy
    self.hist_y = self.widget_histy.addPlot()
    #hide the axes
    self.hist_y.hideAxis('left')
    self.hist_y.setYLink(self.scatter_xy) ## test linking by name
    self.hist_y.showGrid(x=True,y=True)

    #PlotItem for the Fl-trace:
    self.plot_fl_trace = self.widget_showFltrace.addPlot()
    self.plot_fl_trace.showGrid(x=True,y=True)
    
    #When horizontalSlider_cellInd is changed, initiate an onClick event and the points have to be accordigly
    self.horizontalSlider_cellInd.valueChanged.connect(self.onIndexChange)
    self.spinBox_cellInd.valueChanged.connect(self.onIndexChange)

    #Select a peak
    self.pushButton_selectPeakPos.clicked.connect(self.selectPeakPos)
    #Select range
    self.pushButton_selectPeakRange.clicked.connect(self.selectPeakRange)
    #Initiate the peak-table
    #self.tableWidget_showSelectedPeaks.append("FL_max\tFl_pos\tpos_x")
    self.tableWidget_showSelectedPeaks.setColumnCount(0)
    self.tableWidget_showSelectedPeaks.setColumnCount(5)
    self.tableWidget_showSelectedPeaks.setRowCount(0)
    header_labels = ["FL_max","Fl_pos_[us]","pos_x_[um]","Fl_pos_[]","pos_x_[]"]
    self.tableWidget_showSelectedPeaks.setHorizontalHeaderLabels(header_labels) 
    header = self.tableWidget_showSelectedPeaks.horizontalHeader()
    for i in range(3):
        header.setResizeMode(i, QtWidgets.QHeaderView.ResizeToContents)        
    
    
    
    self.pushButton_removeSelectedPeaks.clicked.connect(self.actionRemoveSelectedPeaks_function)
    self.pushButton_removeAllPeaks.clicked.connect(self.actionRemoveAllPeaks_function)

    self.selectedPeaksPlot = self.widget_showSelectedPeaks.addPlot()
    self.selectedPeaksPlot.showGrid(x=True,y=True)
    self.selectedPeaksPlot.setLabel('bottom', 'pos_x', units='um')
    self.selectedPeaksPlot.setLabel('left', 'flx_pos', units='us')

    self.comboBox_peakDetModel.addItems(["Linear dependency and max in range"])
    self.comboBox_peakDetModel.setMinimumSize(QtCore.QSize(200,22))
    self.comboBox_peakDetModel.setMaximumSize(QtCore.QSize(200, 22))
    width=self.comboBox_peakDetModel.fontMetrics().boundingRect(max(self.norm_methods, key=len)).width()
    self.comboBox_peakDetModel.view().setFixedWidth(width+10)             
    self.pushButton_fitPeakDetModel.clicked.connect(self.update_peak_plot)

    self.pushButton_highestXPercent.clicked.connect(self.addHighestXPctPeaks)
    self.pushButton_SavePeakDetModel.clicked.connect(self.savePeakDetModel)
    self.checkBox_fl1.setChecked(True)
    self.checkBox_fl2.setChecked(True)
    self.checkBox_fl3.setChecked(True)
    self.checkBox_centroid.setChecked(True)
    self.doubleSpinBox_highestXPercent.setMinimum(0)
    self.doubleSpinBox_highestXPercent.setMaximum(100)
    self.doubleSpinBox_highestXPercent.setValue(5)

    self.tableWidget_AccPrecSpec.cellClicked.connect(lambda: self.copy_cm_to_clipboard(3))
    self.pushButton_loadPeakDetModel.clicked.connect(self.loadPeakDetModel)
    overwrite_methods = ["Overwrite Fl_max and Fl_pos","Save to userdef"]
    self.comboBox_toFlOrUserdef.addItems(overwrite_methods)
    width=self.comboBox_toFlOrUserdef.fontMetrics().boundingRect(max(overwrite_methods, key=len)).width()
    self.comboBox_toFlOrUserdef.view().setFixedWidth(width+10)             
    self.pushButton_export.clicked.connect(self.applyPeakModel_and_export)
    ex_opt = ["Scores and predictions to Excel sheet","Add predictions to .rtdc file (userdef0)"]
    ex_opt.append("Add pred&scores to .rtdc file (userdef0 to 9)")
    self.comboBox_scoresOrPrediction.addItems(ex_opt)
    self.pushButton_classify.clicked.connect(self.classify)



    #########################Python Editor/Console#########################
    self.pushButton_pythonInRun.clicked.connect(self.pythonInRun)
    self.pushButton_pythonInClear.clicked.connect(self.pythonInClear)
    self.pushButton_pythonSaveAs.clicked.connect(self.pythonInSaveAs)
    self.pushButton_pythonInOpen.clicked.connect(self.pythonInOpen)
    self.pushButton_pythonOutClear.clicked.connect(self.pythonOutClear)
    self.pushButton_pythonOutRun.clicked.connect(self.pythonInRun)


    #############################MenuBar###################################
    self.gridLayout_2.addWidget(self.tabWidget_Modelbuilder, 0, 1, 1, 1)
    self.setCentralWidget(self.centralwidget)
    self.menubar = QtWidgets.QMenuBar(self)
    self.menubar.setGeometry(QtCore.QRect(0, 0, 912, 26))
    self.menubar.setObjectName(_fromUtf8("menubar"))
    self.menuFile = QtWidgets.QMenu(self.menubar)
    self.menuFile.setObjectName(_fromUtf8("menuFile"))
#    self.menuEdit = QtWidgets.QMenu(self.menubar)
#    self.menuEdit.setObjectName(_fromUtf8("menuEdit"))
    
    self.menu_Options = QtWidgets.QMenu(self.menubar)
    self.menu_Options.setObjectName("menu_Options")
    self.menu_Options.setToolTipsVisible(True)

    self.menuLayout = QtWidgets.QMenu(self.menu_Options)
    self.menuLayout.setObjectName("menuLayout")
    self.menuExport = QtWidgets.QMenu(self.menu_Options)
    self.menuExport.setObjectName("menuExport")
    self.menuGPU_Options = QtWidgets.QMenu(self.menu_Options)
    self.menuGPU_Options.setObjectName("menuGPU_Options")
    if gpu_nr<2:
        print("Disabled Multi-GPU Options (Menubar)")
        self.menuGPU_Options.setEnabled(False)
    
    self.actioncpu_merge = QtWidgets.QAction(self)
    self.actioncpu_merge.setCheckable(True)
    self.actioncpu_merge.setChecked(True)
    self.actioncpu_merge.setObjectName("actioncpu_merge")
    self.actioncpu_relocation = QtWidgets.QAction(self)
    self.actioncpu_relocation.setCheckable(True)
    self.actioncpu_relocation.setChecked(False)
    self.actioncpu_relocation.setObjectName("actioncpu_relocation")
    self.actioncpu_weightmerge = QtWidgets.QAction(self)        
    self.actioncpu_weightmerge.setCheckable(True)
    self.actioncpu_weightmerge.setChecked(False)
    self.actioncpu_weightmerge.setObjectName("actioncpu_weightmerge")





    self.menu_Help = QtWidgets.QMenu(self.menubar)
    self.menu_Help.setObjectName("menu_Help")
    self.menu_Help.setToolTipsVisible(True)

    self.actionDocumentation = QtWidgets.QAction(self)
    self.actionDocumentation.triggered.connect(self.actionDocumentation_function)
    self.actionDocumentation.setObjectName(_fromUtf8("actionDocumentation"))

    self.actionTerminology = QtWidgets.QAction(self)
    self.actionTerminology.triggered.connect(self.actionTerminology_function)
    self.actionTerminology.setObjectName(_fromUtf8("actionTerminology"))

    self.menu_Help.addAction(self.actionDocumentation)
    self.menu_Help.addAction(self.actionTerminology)

    self.actionSoftware = QtWidgets.QAction(self)
    self.actionSoftware.triggered.connect(self.actionSoftware_function)
    self.actionSoftware.setObjectName(_fromUtf8("actionSoftware"))
    self.menu_Help.addAction(self.actionSoftware)
    self.actionAbout = QtWidgets.QAction(self)
    self.actionAbout.triggered.connect(self.actionAbout_function)
    self.actionAbout.setObjectName(_fromUtf8("actionAbout"))
    self.menu_Help.addAction(self.actionAbout)
    self.actionUpdate = QtWidgets.QAction(self)
    self.actionUpdate.triggered.connect(self.actionUpdate_check_function)
    self.actionUpdate.setObjectName(_fromUtf8("actionUpdate"))
    self.menu_Help.addAction(self.actionUpdate)



    self.setMenuBar(self.menubar)
    self.statusbar = QtWidgets.QStatusBar(self)
    self.statusbar.setObjectName(_fromUtf8("statusbar"))
    self.setStatusBar(self.statusbar)
    self.statusbar_cpuRam = QtWidgets.QLabel("CPU: xx%  RAM: xx%   ")
    self.statusbar.addPermanentWidget(self.statusbar_cpuRam)        
    
    self.actionLoadSession = QtWidgets.QAction(self)
    self.actionLoadSession.triggered.connect(self.actionLoadSession_function)
    self.actionLoadSession.setObjectName(_fromUtf8("actionLoadSession"))
    self.actionSaveSession = QtWidgets.QAction(self)
    self.actionSaveSession.triggered.connect(self.actionSaveSession_function)
    self.actionSaveSession.setObjectName(_fromUtf8("actionSaveSession"))
    
    self.actionQuit = QtWidgets.QAction(self)
    self.actionQuit.setObjectName(_fromUtf8("actionQuit"))
    self.actionQuit.triggered.connect(self.quit_app)
    
    self.actionDataToRamNow = QtWidgets.QAction(self)
    self.actionDataToRamNow.triggered.connect(self.actionDataToRamNow_function)
    self.actionDataToRamNow.setObjectName(_fromUtf8("actionDataToRamNow"))
    self.actionDataToRam = QtWidgets.QAction(self,checkable=True)
    self.actionDataToRam.setChecked(True)
    self.actionDataToRam.setObjectName(_fromUtf8("actionDataToRam"))

    self.actionKeep_Data_in_RAM = QtWidgets.QAction(self,checkable=True)
    self.actionKeep_Data_in_RAM.setChecked(False)
    self.actionKeep_Data_in_RAM.setObjectName(_fromUtf8("actionKeep_Data_in_RAM"))
    
    self.actionVerbose = QtWidgets.QAction(self,checkable=True)
    self.actionVerbose.setObjectName(_fromUtf8("actionVerbose"))
    self.actionClearList = QtWidgets.QAction(self)
    self.actionClearList.triggered.connect(self.actionClearList_function)
    self.actionClearList.setObjectName(_fromUtf8("actionClearList"))
    self.actionRemoveSelected = QtWidgets.QAction(self)
    self.actionRemoveSelected.setObjectName(_fromUtf8("actionRemoveSelected"))
    self.actionRemoveSelected.triggered.connect(self.actionRemoveSelected_function)
    self.actionSaveToPng = QtWidgets.QAction(self)
    self.actionSaveToPng.setObjectName(_fromUtf8("actionSaveToPng"))
    self.actionSaveToPng.triggered.connect(self.actionSaveToPng_function)
    self.actionClearMemory = QtWidgets.QAction(self)
    self.actionClearMemory.setObjectName(_fromUtf8("actionSaveToPng"))
    self.actionClearMemory.triggered.connect(aid_dl.reset_keras)

    self.actionOpenTemp = QtWidgets.QAction(self)
    self.actionOpenTemp.setObjectName(_fromUtf8("actionOpenTemp"))
    self.actionOpenTemp.triggered.connect(aid_bin.open_temp)

    self.actionGroup_Export = QtWidgets.QActionGroup(self,exclusive=True)
    self.actionExport_Off = QtWidgets.QAction(self)
    self.actionExport_Off.setCheckable(True)
    self.actionExport_Off.setObjectName("actionExport_Off")
    self.actionExport_Original = QtWidgets.QAction(self)
    self.actionExport_Original.setCheckable(True)
    self.actionExport_Original.setChecked(True)
    self.actionExport_Original.setObjectName("actionExport_Original")
    self.actionExport_Cropped = QtWidgets.QAction(self)
    self.actionExport_Cropped.setCheckable(True)
    self.actionExport_Cropped.setChecked(False)
    self.actionExport_Cropped.setObjectName("actionExport_Cropped")
    a = self.actionGroup_Export.addAction(self.actionExport_Off)
    self.menuExport.addAction(a)
    a = self.actionGroup_Export.addAction(self.actionExport_Original)
    self.menuExport.addAction(a)
    a = self.actionGroup_Export.addAction(self.actionExport_Cropped)
    self.menuExport.addAction(a)

    
    self.actionGroup_Layout = QtWidgets.QActionGroup(self,exclusive=True)
    self.actionLayout_Normal = QtWidgets.QAction(self)
    self.actionLayout_Normal.setCheckable(True)
    self.actionLayout_Normal.setObjectName("actionLayout_Normal")
    self.actionLayout_Dark = QtWidgets.QAction(self)
    self.actionLayout_Dark.setCheckable(True)
    self.actionLayout_Dark.setObjectName("actionLayout_Dark")
    self.actionLayout_DarkOrange = QtWidgets.QAction(self)
    self.actionLayout_DarkOrange.setCheckable(True)
    self.actionLayout_DarkOrange.setObjectName("actionLayout_DarkOrange")

    if Default_dict["Layout"] == "Normal":
        self.actionLayout_Normal.setChecked(True)
    elif Default_dict["Layout"] == "Dark":
        self.actionLayout_Dark.setChecked(True)
    elif Default_dict["Layout"] == "DarkOrange":
        self.actionLayout_DarkOrange.setChecked(True)

    a = self.actionGroup_Layout.addAction(self.actionLayout_Normal)
    self.menuLayout.addAction(a)
    a = self.actionGroup_Layout.addAction(self.actionLayout_Dark)
    self.menuLayout.addAction(a)
    a = self.actionGroup_Layout.addAction(self.actionLayout_DarkOrange)
    self.menuLayout.addAction(a)
    self.menuLayout.addSeparator()
    self.actionTooltipOnOff = QtWidgets.QAction(self,checkable=True)
    self.actionTooltipOnOff.setChecked(True)
    self.actionTooltipOnOff.setObjectName(_fromUtf8("actionTooltipOnOff"))
    self.menuLayout.addAction(self.actionTooltipOnOff)        

    self.menuLayout.addSeparator()

    self.actionGroup_IconTheme = QtWidgets.QActionGroup(self,exclusive=True)
    self.actionIconTheme_1 = QtWidgets.QAction(self)
    self.actionIconTheme_1.setCheckable(True)
    self.actionIconTheme_1.setObjectName("actionIconTheme_1")
    self.actionIconTheme_2 = QtWidgets.QAction(self)
    self.actionIconTheme_2.setCheckable(True)
    self.actionIconTheme_2.setObjectName("actionIconTheme_2")

    if Default_dict["Icon theme"] == "Icon theme 1":
        self.actionIconTheme_1.setChecked(True)
    elif Default_dict["Icon theme"] == "Icon theme 2":
        self.actionIconTheme_2.setChecked(True)

    a = self.actionGroup_IconTheme.addAction(self.actionIconTheme_1)
    self.menuLayout.addAction(a)
    a = self.actionGroup_IconTheme.addAction(self.actionIconTheme_2)
    self.menuLayout.addAction(a)

    self.menuGPU_Options.addAction(self.actioncpu_merge)
    self.menuGPU_Options.addAction(self.actioncpu_relocation)
    self.menuGPU_Options.addAction(self.actioncpu_weightmerge)

    self.menu_Options.addAction(self.menuLayout.menuAction())
    self.menu_Options.addSeparator()
    self.menu_Options.addAction(self.menuExport.menuAction())
    self.menu_Options.addSeparator()
    self.menu_Options.addAction(self.actionOpenTemp)
    self.menu_Options.addSeparator()
    self.menu_Options.addAction(self.actionClearMemory)
    self.menu_Options.addSeparator()
    self.menu_Options.addAction(self.menuGPU_Options.menuAction())
    self.menu_Options.addSeparator()
    self.menu_Options.addAction(self.actionVerbose)

    self.menuFile.addAction(self.actionLoadSession)
    self.menuFile.addAction(self.actionSaveSession)
    self.menuFile.addSeparator()
    self.menuFile.addAction(self.actionClearList)
    self.menuFile.addAction(self.actionRemoveSelected)
    self.menuFile.addSeparator()
    self.menuFile.addAction(self.actionDataToRamNow)        
    self.menuFile.addAction(self.actionDataToRam)
    self.menuFile.addAction(self.actionKeep_Data_in_RAM)

    self.menuFile.addAction(self.actionSaveToPng)
    self.menuFile.addSeparator()    
    self.menuFile.addAction(self.actionQuit)


    self.menubar.addAction(self.menuFile.menuAction())
#    self.menubar.addAction(self.menuEdit.menuAction())
    self.menubar.addAction(self.menu_Options.menuAction())
    self.menubar.addAction(self.menu_Help.menuAction())

    #Add Default values:
    self.spinBox_imagecrop.setValue(Default_dict ["Input image size"])
    default_norm = Default_dict["Normalization"]
    index = self.comboBox_Normalization.findText(default_norm, QtCore.Qt.MatchFixedString)
    if index >= 0:
        self.comboBox_Normalization.setCurrentIndex(index)
    self.spinBox_NrEpochs.setValue(Default_dict ["Nr. epochs"])
    self.spinBox_RefreshAfterEpochs.setValue(Default_dict ["Keras refresh after nr. epochs"])
    self.checkBox_HorizFlip.setChecked(Default_dict ["Horz. flip"])
    self.checkBox_VertFlip.setChecked(Default_dict ["Vert. flip"])

    self.lineEdit_Rotation.setText(str(Default_dict ["rotation"]))
 
    self.lineEdit_Rotation.setText(str(Default_dict ["rotation"]))
    self.lineEdit_widthShift.setText(str(Default_dict ["width_shift"]))
    self.lineEdit_heightShift.setText(str(Default_dict ["height_shift"]))
    self.lineEdit_zoomRange.setText(str(Default_dict ["zoom"]))
    self.lineEdit_shearRange.setText(str(Default_dict ["shear"]))
    self.spinBox_RefreshAfterNrEpochs.setValue(Default_dict ["Brightness refresh after nr. epochs"])
    self.spinBox_PlusLower.setValue(Default_dict ["Brightness add. lower"])
    self.spinBox_PlusUpper.setValue(Default_dict ["Brightness add. upper"])
    self.doubleSpinBox_MultLower.setValue(Default_dict ["Brightness mult. lower"])        
    self.doubleSpinBox_MultUpper.setValue(Default_dict ["Brightness mult. upper"])
    self.doubleSpinBox_GaussianNoiseMean.setValue(Default_dict ["Gaussnoise Mean"])
    self.doubleSpinBox_GaussianNoiseScale.setValue(Default_dict ["Gaussnoise Scale"])


    self.checkBox_contrast.setChecked(Default_dict["Contrast On"])
    self.doubleSpinBox_contrastLower.setEnabled(Default_dict["Contrast On"])
    self.doubleSpinBox_contrastHigher.setEnabled(Default_dict["Contrast On"])
    self.doubleSpinBox_contrastLower.setValue(Default_dict["Contrast min"])
    self.doubleSpinBox_contrastHigher.setValue(Default_dict["Contrast max"])
    
    self.checkBox_saturation.setChecked(Default_dict["Saturation On"])
    self.doubleSpinBox_saturationLower.setEnabled(Default_dict["Saturation On"])
    self.doubleSpinBox_saturationHigher.setEnabled(Default_dict["Saturation On"])
    self.doubleSpinBox_saturationLower.setValue(Default_dict["Saturation min"])
    self.doubleSpinBox_saturationHigher.setValue(Default_dict["Saturation max"])

    self.checkBox_hue.setChecked(Default_dict["Hue On"])
    self.doubleSpinBox_hueDelta.setEnabled(Default_dict["Hue On"])
    self.doubleSpinBox_hueDelta.setValue(Default_dict["Hue range"])

    self.checkBox_avgBlur.setChecked(Default_dict["AvgBlur On"])
    self.label_avgBlurMin.setEnabled(Default_dict["AvgBlur On"])
    self.spinBox_avgBlurMin.setEnabled(Default_dict["AvgBlur On"])
    self.label_avgBlurMax.setEnabled(Default_dict["AvgBlur On"])
    self.spinBox_avgBlurMax.setEnabled(Default_dict["AvgBlur On"])
    self.spinBox_avgBlurMin.setValue(Default_dict["AvgBlur min"])
    self.spinBox_avgBlurMax.setValue(Default_dict["AvgBlur max"])

    self.checkBox_gaussBlur.setChecked(Default_dict["GaussBlur On"])
    self.label_gaussBlurMin.setEnabled(Default_dict["GaussBlur On"])
    self.spinBox_gaussBlurMin.setEnabled(Default_dict["GaussBlur On"])
    self.label_gaussBlurMax.setEnabled(Default_dict["GaussBlur On"])
    self.spinBox_gaussBlurMax.setEnabled(Default_dict["GaussBlur On"])
    self.spinBox_gaussBlurMin.setValue(Default_dict["GaussBlur min"])
    self.spinBox_gaussBlurMax.setValue(Default_dict["GaussBlur max"])

    self.checkBox_motionBlur.setChecked(Default_dict["MotionBlur On"])
    self.label_motionBlurKernel.setEnabled(Default_dict["MotionBlur On"])
    self.lineEdit_motionBlurKernel.setEnabled(Default_dict["MotionBlur On"])
    self.label_motionBlurAngle.setEnabled(Default_dict["MotionBlur On"])
    self.lineEdit_motionBlurAngle.setEnabled(Default_dict["MotionBlur On"])
    self.lineEdit_motionBlurKernel.setText(str(Default_dict["MotionBlur Kernel"]))
    self.lineEdit_motionBlurAngle.setText(str(Default_dict["MotionBlur Angle"]))



    self.actionLayout_Normal.triggered.connect(self.onLayoutChange)
    self.actionLayout_Dark.triggered.connect(self.onLayoutChange)
    self.actionLayout_DarkOrange.triggered.connect(self.onLayoutChange)

    self.actionTooltipOnOff.triggered.connect(self.onTooltipOnOff)

    self.actionIconTheme_1.triggered.connect(self.onIconThemeChange)
    self.actionIconTheme_2.triggered.connect(self.onIconThemeChange)

    self.retranslateUi()
    
#        self.checkBox_contrast.clicked['bool'].connect(self.doubleSpinBox_contrastLower.setEnabled)
#        self.checkBox_contrast.clicked['bool'].connect(self.doubleSpinBox_contrastHigher.setEnabled)
    self.checkBox_avgBlur.clicked['bool'].connect(self.spinBox_avgBlurMin.setEnabled)
    self.checkBox_avgBlur.clicked['bool'].connect(self.spinBox_avgBlurMax.setEnabled)
    self.checkBox_gaussBlur.clicked['bool'].connect(self.spinBox_gaussBlurMin.setEnabled)
    self.checkBox_gaussBlur.clicked['bool'].connect(self.spinBox_gaussBlurMax.setEnabled)
    self.checkBox_motionBlur.clicked['bool'].connect(self.label_motionBlurKernel.setEnabled)
    self.checkBox_motionBlur.clicked['bool'].connect(self.lineEdit_motionBlurKernel.setEnabled)
    self.checkBox_motionBlur.clicked['bool'].connect(self.label_motionBlurAngle.setEnabled)
    self.checkBox_motionBlur.clicked['bool'].connect(self.lineEdit_motionBlurAngle.setEnabled)
    self.checkBox_gaussBlur.clicked['bool'].connect(self.label_gaussBlurMin.setEnabled)
    self.checkBox_gaussBlur.clicked['bool'].connect(self.label_gaussBlurMax.setEnabled)
    self.checkBox_avgBlur.clicked['bool'].connect(self.label_avgBlurMin.setEnabled)
    self.checkBox_avgBlur.clicked['bool'].connect(self.label_avgBlurMax.setEnabled)
    self.checkBox_optimizer.toggled['bool'].connect(self.comboBox_optimizer.setEnabled)
    self.checkBox_optimizer.toggled['bool'].connect(self.pushButton_optimizer.setEnabled)
    self.checkBox_expt_loss.toggled['bool'].connect(self.comboBox_expt_loss.setEnabled)

    #Start running show_cpu_ram function and run it all the time
    worker_cpu_ram = Worker(self.cpu_ram_worker)
    self.threadpool.start(worker_cpu_ram)

    self.tabWidget_Modelbuilder.setCurrentIndex(0)
    self.tabWidget_DefineModel.setCurrentIndex(0)
    QtCore.QMetaObject.connectSlotsByName(self)






































def retranslate_main_ui(self,gpu_nr,VERSION):
    self.setWindowTitle(_translate("MainWindow", "AIDeveloper v."+VERSION, None))
    self.groupBox_dragdrop.setTitle(_translate("MainWindow", "Drag and drop data (.rtdc) here", None))
    self.groupBox_dragdrop.setToolTip(_translate("MainWindow", tooltips["groupBox_dragdrop"],None))
    self.groupBox_DataOverview.setTitle(_translate("MainWindow", "Data Overview", None))
    self.groupBox_DataOverview.setToolTip(_translate("MainWindow", tooltips["groupBox_DataOverview"],None))

    self.tab_ExampleImgs.setToolTip(_translate("MainWindow", tooltips["tab_ExampleImgs"],None))
    self.comboBox_ModelSelection.setToolTip(_translate("MainWindow", tooltips["comboBox_ModelSelection"],None))
    self.radioButton_NewModel.setToolTip(_translate("MainWindow",tooltips["radioButton_NewModel"] , None))
    self.radioButton_NewModel.setText(_translate("MainWindow", "New", None))
    self.radioButton_LoadRestartModel.setToolTip(_translate("MainWindow",tooltips["radioButton_LoadRestartModel"] , None))
    self.radioButton_LoadRestartModel.setText(_translate("MainWindow", "Load and restart", None))
    self.radioButton_LoadContinueModel.setToolTip(_translate("MainWindow",tooltips["radioButton_LoadContinueModel"] , None))
    self.radioButton_LoadContinueModel.setText(_translate("MainWindow", "Load and continue", None))
    self.lineEdit_LoadModelPath.setToolTip(_translate("MainWindow",tooltips["lineEdit_LoadModelPath"]  , None))
    
    self.groupBox_imgProc.setTitle(_translate("MainWindow", "Image processing", None))
    self.label_CropIcon.setToolTip(_translate("MainWindow",tooltips["label_Crop"] , None))
    self.label_Crop.setToolTip(_translate("MainWindow",tooltips["label_Crop"] , None))
    self.label_Crop.setText(_translate("MainWindow", "<html><head/><body><p>Input image size</p></body></html>", None))
    self.spinBox_imagecrop.setToolTip(_translate("MainWindow",tooltips["label_Crop"] , None))

    self.label_padIcon.setToolTip(_translate("MainWindow", tooltips["label_paddingMode"], None))
    self.label_paddingMode.setText(_translate("MainWindow", "Padding mode", None))
    self.label_paddingMode.setToolTip(_translate("MainWindow", tooltips["label_paddingMode"], None))
    self.comboBox_paddingMode.setToolTip(_translate("MainWindow", tooltips["label_paddingMode"], None))
    self.label_zoomIcon.setToolTip(_translate("MainWindow", tooltips["label_zoomIcon"], None))
    self.label_zoomOrder.setText(_translate("MainWindow", "Zoom order", None))
    self.label_zoomOrder.setToolTip(_translate("MainWindow", tooltips["label_zoomIcon"], None))


    self.comboBox_paddingMode.setItemText(0, _translate("MainWindow", "constant", None))
    self.comboBox_paddingMode.setItemText(1, _translate("MainWindow", "edge", None))
    self.comboBox_paddingMode.setItemText(2, _translate("MainWindow", "reflect", None))
    self.comboBox_paddingMode.setItemText(3, _translate("MainWindow", "symmetric", None))
    self.comboBox_paddingMode.setItemText(4, _translate("MainWindow", "wrap", None))
    self.comboBox_paddingMode.setItemText(5, _translate("MainWindow", "delete", None))
    self.comboBox_paddingMode.setItemText(6, _translate("MainWindow", "alternate", None))

    self.comboBox_zoomOrder.setItemText(0, _translate("MainWindow", "nearest neighbor (cv2.INTER_NEAREST)", None))
    self.comboBox_zoomOrder.setItemText(1, _translate("MainWindow", "lin. interp. (cv2.INTER_LINEAR)", None))
    self.comboBox_zoomOrder.setItemText(2, _translate("MainWindow", "quadr. interp. (cv2.INTER_AREA)", None))
    self.comboBox_zoomOrder.setItemText(3, _translate("MainWindow", "cubic interp. (cv2.INTER_CUBIC)", None))
    self.comboBox_zoomOrder.setItemText(4, _translate("MainWindow", "Lanczos 4 (cv2.INTER_LANCZOS4)", None))
    zoomitems = [self.comboBox_zoomOrder.itemText(i) for i in range(self.comboBox_zoomOrder.count())]
    width=self.comboBox_zoomOrder.fontMetrics().boundingRect(max(zoomitems, key=len)).width()
    self.comboBox_zoomOrder.view().setFixedWidth(width+10)             


    self.label_NormalizationIcon.setToolTip(_translate("MainWindow", tooltips["label_Normalization"], None))        
    self.label_Normalization.setToolTip(_translate("MainWindow", tooltips["label_Normalization"], None))
    self.label_Normalization.setText(_translate("MainWindow", "Normalization", None))
    self.comboBox_Normalization.setToolTip(_translate("MainWindow", tooltips["label_Normalization"], None))

    self.label_colorModeIcon.setToolTip(_translate("MainWindow", tooltips["label_colorMode"], None))
    self.label_colorMode.setText(_translate("MainWindow", "Color Mode", None))
    self.label_colorMode.setToolTip(_translate("MainWindow", tooltips["label_colorMode"],None))    
    self.comboBox_GrayOrRGB.setToolTip(_translate("MainWindow", tooltips["label_colorMode"],None))


    self.groupBox_system.setTitle(_translate("MainWindow", "Training", None))

    #Add CPU elements to dropdown menu
    self.comboBox_cpu.addItem("")
    self.comboBox_cpu.setItemText(0, _translate("MainWindow", "Default CPU", None))

#        for i in range(len(devices_cpu)):
#            self.comboBox_cpu.addItem("")
#            if len(devices_cpu[i].physical_device_desc)==0:
#                self.comboBox_cpu.setItemText(i, _translate("MainWindow", "Default CPU", None))
#            else:
#                self.comboBox_cpu.setItemText(i, _translate("MainWindow", devices_cpu[i].physical_device_desc, None))
            
    if gpu_nr==0:
        self.comboBox_gpu.addItem("")
        self.comboBox_gpu.setItemText(0, _translate("MainWindow", "None", None))
        self.radioButton_gpu.setEnabled(False)
        self.radioButton_cpu.setChecked(True)
    elif gpu_nr==1:
        self.comboBox_gpu.addItem("")
        self.comboBox_gpu.setItemText(0, _translate("MainWindow", "Single-GPU", None))
        self.radioButton_gpu.setChecked(True)#If GPU available, use it by default.

    else: #nr_gpu>1
        self.comboBox_gpu.addItem("")
        self.comboBox_gpu.addItem("")
        self.comboBox_gpu.setItemText(0, _translate("MainWindow", "Single-GPU", None))
        self.comboBox_gpu.setItemText(1, _translate("MainWindow", "Multi-GPU", None))
        self.radioButton_gpu.setChecked(True)#If GPU available, use it by default.

#            for i in range(len(devices_gpu)):
#                self.comboBox_gpu.addItem("")
#                self.comboBox_gpu.setItemText(i, _translate("MainWindow", devices_gpu[i].physical_device_desc, None))

    self.label_memory.setText(_translate("MainWindow", "Memory", None))
    self.label_memory.setToolTip(_translate("MainWindow", tooltips["label_memory"],None))
    self.doubleSpinBox_memory.setToolTip(_translate("MainWindow", tooltips["label_memory"],None))

    self.label_nrEpochsIcon.setToolTip(_translate("MainWindow",tooltips["label_nrEpochs"] , None))
    self.label_nrEpochs.setToolTip(_translate("MainWindow",tooltips["label_nrEpochs"] , None))
    self.label_nrEpochs.setText(_translate("MainWindow", "Nr. epochs", None))
    self.spinBox_NrEpochs.setToolTip(_translate("MainWindow",tooltips["label_nrEpochs"] , None))
    self.radioButton_cpu.setToolTip(_translate("MainWindow",tooltips["radioButton_cpu"] , None))
    self.comboBox_cpu.setToolTip(_translate("MainWindow",tooltips["radioButton_cpu"] , None))
    self.radioButton_gpu.setToolTip(_translate("MainWindow",tooltips["radioButton_gpu"] , None))
    self.comboBox_gpu.setToolTip(_translate("MainWindow",tooltips["comboBox_gpu"] , None))

    self.pushButton_modelname.setToolTip(_translate("MainWindow", tooltips["pushButton_modelname"], None))
    self.pushButton_modelname.setText(_translate("MainWindow", "Model path:", None))
    self.lineEdit_modelname.setToolTip(_translate("MainWindow", tooltips["pushButton_modelname"], None))

    self.tabWidget_DefineModel.setTabText(self.tabWidget_DefineModel.indexOf(self.tab_DefineModel), _translate("MainWindow", "Define Model", None))
    self.label_RefreshAfterEpochs.setToolTip(_translate("MainWindow", tooltips["label_RefreshAfterEpochs"], None))
    self.label_RefreshAfterEpochs.setText(_translate("MainWindow", "Refresh after nr. epochs:", None))
    self.tab_kerasAug.setToolTip(_translate("MainWindow",tooltips["tab_kerasAug"] , None))
    self.checkBox_HorizFlip.setToolTip(_translate("MainWindow",tooltips["checkBox_HorizFlip"] , None))
    self.checkBox_VertFlip.setToolTip(_translate("MainWindow",tooltips["checkBox_VertFlip"] , None))
    self.label_Rotation.setToolTip(_translate("MainWindow",tooltips["label_Rotation"] , None))
    self.label_width_shift.setToolTip(_translate("MainWindow",tooltips["label_width_shift"] , None))
    self.label_height_shift.setToolTip(_translate("MainWindow",tooltips["label_height_shift"], None))
    self.label_zoom.setToolTip(_translate("MainWindow",tooltips["label_zoom"] , None))
    self.label_shear.setToolTip(_translate("MainWindow",tooltips["label_shear"] , None))        
    self.lineEdit_Rotation.setToolTip(_translate("MainWindow",tooltips["label_Rotation"] , None))
    self.lineEdit_widthShift.setToolTip(_translate("MainWindow",tooltips["label_width_shift"] , None))
    self.lineEdit_heightShift.setToolTip(_translate("MainWindow",tooltips["label_height_shift"] , None))
    self.lineEdit_zoomRange.setToolTip(_translate("MainWindow", tooltips["label_zoom"], None))
    self.lineEdit_shearRange.setToolTip(_translate("MainWindow", tooltips["label_shear"], None))
    self.spinBox_RefreshAfterEpochs.setToolTip(_translate("MainWindow",tooltips["spinBox_RefreshAfterEpochs"] , None))
    self.checkBox_HorizFlip.setText(_translate("MainWindow", "Horiz. flip", None))
    self.checkBox_VertFlip.setText(_translate("MainWindow", "Vert. flip", None))
    self.label_Rotation.setText(_translate("MainWindow", "Rotation", None))
    self.label_width_shift.setText(_translate("MainWindow", "Width shift", None))
    self.label_height_shift.setText(_translate("MainWindow", "Height shift", None))
    self.label_zoom.setText(_translate("MainWindow", "Zoom", None))
    self.label_shear.setText(_translate("MainWindow", "Shear", None))
    self.tabWidget_DefineModel.setTabText(self.tabWidget_DefineModel.indexOf(self.tab_kerasAug), _translate("MainWindow", "Affine img. augm.", None))
    self.label_RefreshAfterNrEpochs.setToolTip(_translate("MainWindow",tooltips["label_RefreshAfterNrEpochs"] , None))
    self.label_RefreshAfterNrEpochs.setText(_translate("MainWindow", "Refresh after nr. epochs:", None))
    self.spinBox_RefreshAfterNrEpochs.setToolTip(_translate("MainWindow", tooltips["label_RefreshAfterNrEpochs"], None))
    self.groupBox_BrightnessAugmentation.setTitle(_translate("MainWindow", "Brightness augmentation", None))

    self.groupBox_BrightnessAugmentation.setToolTip(_translate("MainWindow",tooltips["groupBox_BrightnessAugmentation"] , None))

    self.label_Plus.setText(_translate("MainWindow", "Add.", None))
    self.label_Plus.setToolTip(_translate("MainWindow",tooltips["label_Plus"] , None))
    self.spinBox_PlusLower.setToolTip(_translate("MainWindow",tooltips["spinBox_PlusLower"] , None))
    self.spinBox_PlusUpper.setToolTip(_translate("MainWindow",tooltips["spinBox_PlusUpper"] , None))

    self.label_Mult.setText(_translate("MainWindow", "Mult.", None))
    self.label_Mult.setToolTip(_translate("MainWindow",tooltips["label_Mult"] , None))

    self.doubleSpinBox_MultLower.setToolTip(_translate("MainWindow",tooltips["doubleSpinBox_MultLower"] , None))
    self.doubleSpinBox_MultUpper.setToolTip(_translate("MainWindow", tooltips["doubleSpinBox_MultUpper"], None))

    #self.label_Rotation_MultTo.setText(_translate("MainWindow", "...", None))
    self.groupBox_GaussianNoise.setTitle(_translate("MainWindow", "Gaussian noise", None))
    self.groupBox_GaussianNoise.setToolTip(_translate("MainWindow",tooltips["groupBox_GaussianNoise"] , None))
    self.label_GaussianNoiseMean.setText(_translate("MainWindow", "Mean", None))
    self.label_GaussianNoiseMean.setToolTip(_translate("MainWindow",tooltips["label_GaussianNoiseMean"]  , None))

    self.label_GaussianNoiseScale.setText(_translate("MainWindow", "Scale", None))
    self.label_GaussianNoiseScale.setToolTip(_translate("MainWindow",tooltips["label_GaussianNoiseScale"] , None))

    self.groupBox_colorAugmentation.setTitle(_translate("Form", "Color augm.", None))
    self.groupBox_colorAugmentation.setToolTip(_translate("MainWindow",tooltips["groupBox_colorAugmentation"] , None))
    self.checkBox_contrast.setText(_translate("Form", "Contrast", None))
    self.checkBox_contrast.setToolTip(_translate("Form",tooltips["checkBox_contrast"] , None))
    self.checkBox_saturation.setText(_translate("Form", "Saturation", None))
    self.checkBox_saturation.setToolTip(_translate("Form",tooltips["checkBox_saturation"] , None))
    self.checkBox_hue.setText(_translate("Form", "Hue", None))
    self.checkBox_hue.setToolTip(_translate("Form",tooltips["checkBox_hue"] , None))

    self.groupBox_blurringAug.setTitle(_translate("MainWindow", "Blurring", None))
    self.groupBox_blurringAug.setToolTip(_translate("MainWindow",tooltips["groupBox_blurringAug"] , None))
    self.label_motionBlurKernel.setToolTip(_translate("MainWindow",tooltips["label_motionBlurKernel"]  , None))
    self.label_motionBlurKernel.setText(_translate("MainWindow", "Kernel", None))
    self.lineEdit_motionBlurAngle.setToolTip(_translate("MainWindow",tooltips["lineEdit_motionBlurAngle"] , None))
    self.label_avgBlurMin.setToolTip(_translate("MainWindow",tooltips["label_avgBlurMin"] , None))
    self.label_avgBlurMin.setText(_translate("MainWindow", "Min", None))
    self.spinBox_gaussBlurMax.setToolTip(_translate("MainWindow", tooltips["spinBox_gaussBlurMax"], None))
    self.checkBox_motionBlur.setToolTip(_translate("MainWindow",tooltips["checkBox_motionBlur"] , None))
    self.checkBox_motionBlur.setText(_translate("MainWindow", "Motion", None))
    self.spinBox_avgBlurMin.setToolTip(_translate("MainWindow",tooltips["spinBox_avgBlurMin"] , None))
    self.spinBox_gaussBlurMin.setToolTip(_translate("MainWindow", tooltips["spinBox_gaussBlurMin"], None))
    self.label_motionBlurAngle.setToolTip(_translate("MainWindow", tooltips["label_motionBlurAngle"], None))
    self.label_motionBlurAngle.setText(_translate("MainWindow", "Angle", None))
    self.label_gaussBlurMin.setToolTip(_translate("MainWindow",tooltips["label_gaussBlurMin"] , None))
    self.label_gaussBlurMin.setText(_translate("MainWindow", "Min", None))
    self.checkBox_gaussBlur.setToolTip(_translate("MainWindow",tooltips["checkBox_gaussBlur"] , None))
    self.checkBox_gaussBlur.setText(_translate("MainWindow", "Gauss", None))
    self.spinBox_avgBlurMax.setToolTip(_translate("MainWindow",tooltips["spinBox_avgBlurMax"] , None))
    self.label_gaussBlurMax.setToolTip(_translate("MainWindow",tooltips["label_gaussBlurMax"] , None))
    self.label_gaussBlurMax.setText(_translate("MainWindow", "Max", None))
    self.label_gaussBlurMax.setToolTip(_translate("MainWindow", tooltips["label_gaussBlurMax"], None))
    self.checkBox_avgBlur.setToolTip(_translate("MainWindow",tooltips["checkBox_avgBlur"] , None))        
    self.checkBox_avgBlur.setText(_translate("MainWindow", "Average", None))
    
    self.label_avgBlurMax.setToolTip(_translate("MainWindow",tooltips["label_avgBlurMax"] , None))
    self.label_avgBlurMax.setText(_translate("MainWindow", "Max", None))
    self.spinBox_avgBlurMin.setToolTip(_translate("MainWindow",tooltips["spinBox_avgBlurMin"] , None))
    self.spinBox_avgBlurMax.setToolTip(_translate("MainWindow", tooltips["spinBox_avgBlurMax"], None))
    self.lineEdit_motionBlurKernel.setToolTip(_translate("MainWindow",tooltips["lineEdit_motionBlurKernel"] , None))

    self.tabWidget_DefineModel.setTabText(self.tabWidget_DefineModel.indexOf(self.tab_BrightnessAug), _translate("MainWindow", "Brightn/Color augm.", None))
    self.label_ShowIndex.setText(_translate("MainWindow", "Class", None))
    self.pushButton_ShowExamleImgs.setText(_translate("MainWindow", "Show", None))
    self.tabWidget_DefineModel.setTabText(self.tabWidget_DefineModel.indexOf(self.tab_ExampleImgs), _translate("MainWindow", "Example imgs.", None))
    
    self.groupBox_expertMode.setTitle(_translate("MainWindow", "Expert Mode", None))
    self.groupBox_modelKerasFit.setTitle(_translate("MainWindow", "In model.fit()", None))
    self.groupBox_regularization.setTitle(_translate("MainWindow", "Regularization", None))

    self.label_batchSize.setText(_translate("MainWindow", "Batch size", None))
    self.label_epochs.setText(_translate("MainWindow", "Epochs", None))
    self.pushButton_LR_finder.setText(_translate("MainWindow", "LR Screen", None))
    self.pushButton_LR_finder.setToolTip(_translate("MainWindow", "Screen learning rates. Function disabled on Fitting screen. Please use main screen.", None))    
    self.pushButton_LR_finder.clicked.connect(self.popup_lr_finder)

    self.pushButton_LR_plot.setText(_translate("MainWindow", "Plot", None))
    self.pushButton_LR_plot.setToolTip(_translate("MainWindow", tooltips["pushButton_LR_plot"], None))
    self.pushButton_LR_plot.clicked.connect(lambda: self.popup_lr_plot(-1))  
    self.pushButton_cycLrPopup.clicked.connect(lambda: self.popup_clr_settings(-1))
    
    
    self.groupBox_lossOptimizer.setTitle(_translate("MainWindow", "Loss / Optimizer", None))
    self.groupBox_lossOptimizer.setToolTip(_translate("MainWindow", tooltips["groupBox_lossOptimizer"], None))

    #self.label_others.setText(_translate("MainWindow", "Others", None))
    self.groupBox_learningRate.setTitle(_translate("MainWindow", "Learning rate (LR)", None))
    self.checkBox_trainLastNOnly.setText(_translate("MainWindow", "Train only last N layers", None))
    self.checkBox_trainDenseOnly.setText(_translate("MainWindow", "Train only Dense layers", None))
    self.checkBox_dropout.setText(_translate("MainWindow", "Dropout", None))
    self.tabWidget_DefineModel.setTabText(self.tabWidget_DefineModel.indexOf(self.tab_expert), _translate("MainWindow", "Expert", None))


    self.groupBox_expertMode.setToolTip(_translate("MainWindow",tooltips["groupBox_expertMode"] , None))
    self.groupBox_learningRate.setToolTip(_translate("MainWindow",tooltips["groupBox_learningRate"] , None))
    self.doubleSpinBox_learningRate.setToolTip(_translate("MainWindow",tooltips["groupBox_learningRate"] , None))
    self.checkBox_optimizer.setText(_translate("MainWindow", "Optimizer", None))
    self.checkBox_optimizer.setToolTip(_translate("MainWindow", tooltips["label_optimizer"] , None))

    self.comboBox_optimizer.setItemText(0, _translate("MainWindow", "Adam", None))
    self.comboBox_optimizer.setItemText(1, _translate("MainWindow", "SGD", None))
    self.comboBox_optimizer.setItemText(2, _translate("MainWindow", "RMSprop", None))
    self.comboBox_optimizer.setItemText(3, _translate("MainWindow", "Adagrad", None))
    self.comboBox_optimizer.setItemText(4, _translate("MainWindow", "Adadelta", None))
    self.comboBox_optimizer.setItemText(5, _translate("MainWindow", "Adamax", None))
    self.comboBox_optimizer.setItemText(6, _translate("MainWindow", "Nadam", None))
    self.comboBox_optimizer.setToolTip(_translate("MainWindow", tooltips["label_optimizer"] , None))
    
    self.pushButton_optimizer.setText(_translate("MainWindow", "...", None))
    self.pushButton_optimizer.setToolTip(_translate("MainWindow", "Show advanced options for optimizer", None))

    self.checkBox_trainLastNOnly.setToolTip(_translate("MainWindow",tooltips["checkBox_trainLastNOnly"] , None))
    self.spinBox_trainLastNOnly.setToolTip(_translate("MainWindow", tooltips["spinBox_trainLastNOnly"], None))
    self.checkBox_trainDenseOnly.setToolTip(_translate("MainWindow",tooltips["checkBox_trainDenseOnly"] , None))
    self.label_batchSize.setToolTip(_translate("MainWindow",tooltips["label_batchSize"] , None))
    self.spinBox_batchSize.setToolTip(_translate("MainWindow",tooltips["label_batchSize"] , None))
    self.label_epochs.setToolTip(_translate("MainWindow",tooltips["label_epochs"] , None))
    self.spinBox_epochs.setToolTip(_translate("MainWindow",tooltips["label_epochs"] , None))

    self.groupBox_learningRate.setTitle(_translate("MainWindow", "Learning rate (LR)", None))
    self.radioButton_LrConst.setText(_translate("MainWindow", "Constant", None))
    self.label_LrConst.setText(_translate("MainWindow", "LR", None))
    self.radioButton_LrCycl.setText(_translate("MainWindow", "Cyclical", None))
    self.label_cycLrMin.setText(_translate("MainWindow", "Range", None))
#        self.label_cycLrMax.setText(_translate("MainWindow", "Max", None))
    self.comboBox_cycLrMethod.setItemText(0, _translate("MainWindow", "triangular", None))
    self.comboBox_cycLrMethod.setItemText(1, _translate("MainWindow", "triangular2", None))
    self.comboBox_cycLrMethod.setItemText(2, _translate("MainWindow", "exp_range", None))
    self.label_cycLrMethod.setText(_translate("MainWindow", "Method", None))
    self.label_cycLrMethod.setToolTip(_translate("MainWindow", tooltips["comboBox_cycLrMethod"], None))
    
    self.pushButton_cycLrPopup.setText(_translate("MainWindow", "...", None))
    self.radioButton_LrExpo.setText(_translate("MainWindow", "Expo.", None))
    
    self.radioButton_LrConst.setToolTip(_translate("MainWindow", tooltips["doubleSpinBox_learningRate"],None))
    self.doubleSpinBox_learningRate.setToolTip(_translate("MainWindow", tooltips["doubleSpinBox_learningRate"],None))
    self.radioButton_LrCycl.setToolTip(_translate("MainWindow", tooltips["radioButton_LrCycl"],None))
    self.label_cycLrMin.setToolTip(_translate("MainWindow", tooltips["label_cycLrMin"],None))
    self.lineEdit_cycLrMin.setToolTip(_translate("MainWindow", tooltips["label_cycLrMin"],None))
    #self.label_cycLrMax.setToolTip(_translate("MainWindow", tooltips["label_cycLrMax"],None))
    self.lineEdit_cycLrMax.setToolTip(_translate("MainWindow", tooltips["label_cycLrMax"],None))
    self.comboBox_cycLrMethod.setToolTip(_translate("MainWindow", tooltips["comboBox_cycLrMethod"],None))
    self.pushButton_cycLrPopup.setToolTip(_translate("MainWindow", tooltips["pushButton_cycLrPopup"],None))
    self.radioButton_LrExpo.setToolTip(_translate("MainWindow", tooltips["radioButton_LrExpo"],None))
    self.label_expDecInitLr.setToolTip(_translate("MainWindow", tooltips["label_expDecInitLr"],None))
    self.doubleSpinBox_expDecInitLr.setToolTip(_translate("MainWindow", tooltips["radioButton_LrExpo"],None))
    self.label_expDecSteps.setToolTip(_translate("MainWindow", tooltips["label_expDecSteps"],None))
    self.spinBox_expDecSteps.setToolTip(_translate("MainWindow", tooltips["label_expDecSteps"],None))
    self.label_expDecRate.setToolTip(_translate("MainWindow", tooltips["label_expDecRate"],None))
    self.doubleSpinBox_expDecRate.setToolTip(_translate("MainWindow", tooltips["label_expDecRate"],None))
    
    
    self.label_expDecInitLr.setText(_translate("MainWindow", "Initial LR",None))
    self.label_expDecSteps.setText(_translate("MainWindow", "Decay steps",None))
    self.label_expDecRate.setText(_translate("MainWindow", "Decay rate",None))


    self.checkBox_expt_loss.setText(_translate("MainWindow", "Loss", None))
    self.checkBox_expt_loss.setToolTip(_translate("MainWindow", tooltips["label_expt_loss"], None))
    self.comboBox_expt_loss.setToolTip(_translate("MainWindow", tooltips["label_expt_loss"], None))

    self.comboBox_expt_loss.setItemText(0, _translate("MainWindow", "categorical_crossentropy", None))
    #self.comboBox_expt_loss.setItemText(1, _translate("MainWindow", "sparse_categorical_crossentropy", None))
    self.comboBox_expt_loss.setItemText(1, _translate("MainWindow", "mean_squared_error", None))
    self.comboBox_expt_loss.setItemText(2, _translate("MainWindow", "mean_absolute_error", None))
    self.comboBox_expt_loss.setItemText(3, _translate("MainWindow", "mean_absolute_percentage_error", None))
    self.comboBox_expt_loss.setItemText(4, _translate("MainWindow", "mean_squared_logarithmic_error", None))
    self.comboBox_expt_loss.setItemText(5, _translate("MainWindow", "squared_hinge", None))
    self.comboBox_expt_loss.setItemText(6, _translate("MainWindow", "hinge", None))
    self.comboBox_expt_loss.setItemText(7, _translate("MainWindow", "categorical_hinge", None))
    self.comboBox_expt_loss.setItemText(8, _translate("MainWindow", "logcosh", None))
    #self.comboBox_expt_loss.setItemText(9, _translate("MainWindow", "huber_loss", None))
    self.comboBox_expt_loss.setItemText(9, _translate("MainWindow", "binary_crossentropy", None))
    self.comboBox_expt_loss.setItemText(10, _translate("MainWindow", "kullback_leibler_divergence", None))
    self.comboBox_expt_loss.setItemText(11, _translate("MainWindow", "poisson", None))
    self.comboBox_expt_loss.setItemText(12, _translate("MainWindow", "cosine_proximity", None))
    #self.comboBox_expt_loss.setItemText(13, _translate("MainWindow", "is_categorical_crossentropy", None))
    
    self.checkBox_dropout.setToolTip(_translate("MainWindow",tooltips["checkBox_dropout"] , None))
    self.lineEdit_dropout.setToolTip(_translate("MainWindow", tooltips["checkBox_dropout"], None))
    self.checkBox_partialTrainability.setText(_translate("MainWindow", "Partial trainablility", None))
    self.checkBox_partialTrainability.setToolTip(_translate("MainWindow",tooltips["checkBox_partialTrainability"] , None))
    self.lineEdit_partialTrainability.setToolTip(_translate("MainWindow", tooltips["checkBox_partialTrainability"], None))
    self.pushButton_partialTrainability.setText(_translate("MainWindow", "...", None))
    self.pushButton_partialTrainability.setToolTip(_translate("MainWindow", tooltips["checkBox_partialTrainability"], None))

    self.checkBox_lossW.setText(_translate("MainWindow", "Loss weights", None))
    self.checkBox_lossW.setToolTip(_translate("MainWindow",tooltips["checkBox_lossW"] , None))
    self.lineEdit_lossW.setToolTip(_translate("MainWindow", tooltips["checkBox_lossW"], None))
    self.pushButton_lossW.setText(_translate("MainWindow", "...", None))



    self.groupBox_expertMetrics.setTitle(_translate("MainWindow", "Metrics", None))
    self.groupBox_expertMetrics.setToolTip(_translate("MainWindow",tooltips["groupBox_expertMetrics"] , None))

    self.checkBox_expertAccuracy.setText(_translate("MainWindow", "Accuracy", None))
    self.checkBox_expertAccuracy.setToolTip(_translate("MainWindow",tooltips["checkBox_expertAccuracy"] , None))
    
    self.checkBox_expertF1.setText(_translate("MainWindow", "F1 score", None))
    self.checkBox_expertF1.setToolTip(_translate("MainWindow",tooltips["checkBox_expertF1"],  None))
    
    self.checkBox_expertPrecision.setText(_translate("MainWindow", "Precision", None))
    self.checkBox_expertPrecision.setToolTip(_translate("MainWindow", tooltips["checkBox_expertPrecision"], None))
    
    self.checkBox_expertRecall.setText(_translate("MainWindow", "Recall", None))
    self.checkBox_expertRecall.setToolTip(_translate("MainWindow", tooltips["checkBox_expertRecall"], None))

    self.groupBox_Finalize.setTitle(_translate("MainWindow", "Model summary and Fit", None))
    self.pushButton_FitModel.setText(_translate("MainWindow", "Initialize/Fit\nModel", None))
    self.pushButton_FitModel.setToolTip(_translate("MainWindow",tooltips["pushButton_FitModel"] , None))

    self.tabWidget_Modelbuilder.setTabText(self.tabWidget_Modelbuilder.indexOf(self.tab_Build), _translate("MainWindow", "Build", None))
    self.pushButton_Live.setToolTip(_translate("MainWindow",tooltips["pushButton_Live"] , None))
    self.pushButton_Live.setText(_translate("MainWindow", "Live!", None))
    self.pushButton_LoadHistory.setToolTip(_translate("MainWindow", tooltips["pushButton_LoadHistory"], None))
    self.pushButton_LoadHistory.setText(_translate("MainWindow", "Load History", None))
    self.lineEdit_LoadHistory.setToolTip(_translate("MainWindow",tooltips["lineEdit_LoadHistory"] , None))
    self.tableWidget_HistoryItems.setToolTip(_translate("MainWindow", tooltips["tableWidget_HistoryItems"], None))
    self.pushButton_UpdateHistoryPlot.setText(_translate("MainWindow", "Update plot", None))
    self.checkBox_rollingMedian.setText(_translate("MainWindow", "Rolling Median", None))
    self.checkBox_rollingMedian.setToolTip(_translate("MainWindow",tooltips["checkBox_rollingMedian"] , None))
    self.horizontalSlider_rollmedi.setToolTip(_translate("MainWindow", tooltips["horizontalSlider_rollmedi"], None))
    
    self.checkBox_linearFit.setText(_translate("MainWindow", "Linear Fit", None))
    self.checkBox_linearFit.setToolTip(_translate("MainWindow",tooltips["checkBox_linearFit"] , None))

    self.pushButton_LoadModel.setText(_translate("MainWindow", "Load model", None))
    
    self.pushButton_LoadModel.setToolTip(_translate("MainWindow", tooltips["pushButton_LoadModel"] , None))

    self.pushButton_convertModel.setText(_translate("MainWindow", "Convert", None))
    self.pushButton_convertModel.setToolTip(_translate("MainWindow",tooltips["pushButton_convertModel"] , None))
    self.tabWidget_Modelbuilder.setTabText(self.tabWidget_Modelbuilder.indexOf(self.tab_History), _translate("MainWindow", "History", None))

    self.groupBox_loadModel.setTitle(_translate("MainWindow", "Load Model", None))
    self.label_ModelIndex_2.setText(_translate("MainWindow", "Model index", None))
    self.lineEdit_ModelSelection_2.setToolTip(_translate("MainWindow",tooltips["lineEdit_ModelSelection_2"] , None))
    self.label_Normalization_2.setText(_translate("MainWindow", "Normalization", None))
    self.label_Crop_2.setText(_translate("MainWindow", "Input size", None))
    self.label_OutClasses_2.setText(_translate("MainWindow", "Output Nr. of classes", None))
    self.pushButton_LoadModel_2.setText(_translate("MainWindow", "Load model", None))
    self.lineEdit_LoadModel_2.setToolTip(_translate("MainWindow", tooltips["lineEdit_LoadModel_2"], None))
    self.tableWidget_Info_2.setToolTip(_translate("MainWindow",tooltips["tableWidget_Info_2"] , None))

    self.pushButton_ExportValidToNpy.setToolTip(_translate("MainWindow",tooltips["pushButton_ExportValidToNpy"] , None))
    self.groupBox_validData.setTitle(_translate("MainWindow", "Data", None))
    self.pushButton_ExportValidToNpy.setText(_translate("MainWindow", "To .rtdc", None))
    self.pushButton_ImportValidFromNpy.setText(_translate("MainWindow", "From .rtdc",None))
    self.pushButton_ImportValidFromNpy.setToolTip(_translate("MainWindow",tooltips["pushButton_ImportValidFromNpy"] , None))

    self.groupBox_settings.setTitle(_translate("MainWindow", "Settings", None))
    self.groupBox_InferenceTime.setTitle(_translate("MainWindow", "Inference time", None))
    self.groupBox_InferenceTime.setToolTip(_translate("MainWindow",tooltips["groupBox_InferenceTime"] , None))
  
    self.pushButton_CompInfTime.setText(_translate("MainWindow", "Compute for N imgs", None))
    self.groupBox_classify.setTitle(_translate("MainWindow", "Classify unlabeled data", None))
    self.pushButton_classify.setText(_translate("MainWindow", "Classify", None))
    self.radioButton_selectAll.setText(_translate("MainWindow", "all", None))
    self.groupBox_settings.setTitle(_translate("MainWindow", "Settings", None))
    
    
    self.groupBox_settings.setTitle(_translate("MainWindow", "Settings", None))
    self.label_SortingIndex.setText(_translate("MainWindow", "Sorting class", None))
    self.label_SortingIndex.setToolTip(_translate("MainWindow",tooltips["label_SortingIndex"] , None))
    self.checkBox_SortingThresh.setText(_translate("MainWindow", "Sorting threshold", None))
    self.checkBox_SortingThresh.setToolTip(_translate("MainWindow",tooltips["checkBox_SortingThresh"] , None))

    self.pushButton_AssessModel.setText(_translate("MainWindow", "Update Plots", None))
    
    self.comboBox_probability_histogram.setToolTip(_translate("MainWindow",tooltips["comboBox_probability_histogram"] ,None))
    
    self.groupBox_3rdPlotSettings.setTitle(_translate("MainWindow", "3rd plot settings", None))
    
    self.groupBox_3rdPlot.setToolTip(_translate("MainWindow", tooltips["groupBox_3rdPlot"], None))

    self.label_3rdPlot.setText(_translate("MainWindow", "What to plot", None))
    self.comboBox_3rdPlot.setToolTip(_translate("MainWindow",tooltips["comboBox_3rdPlot"] , None))
    self.label_Indx1.setToolTip(_translate("MainWindow",tooltips["label_Indx1"] , None))
    self.label_Indx1.setText(_translate("MainWindow", "Indx1", None))
    self.spinBox_Indx1.setToolTip(_translate("MainWindow", tooltips["label_Indx1"], None))
    self.label_Indx2.setToolTip(_translate("MainWindow", tooltips["label_Indx1"], None))
    self.label_Indx2.setText(_translate("MainWindow", "Indx2", None))
    self.spinBox_Indx2.setToolTip(_translate("MainWindow", tooltips["label_Indx1"], None))
    self.groupBox_confusionMatrixPlot.setTitle(_translate("MainWindow", "Classification Metrics", None))
    self.tableWidget_CM1.setToolTip(_translate("MainWindow",tooltips["tableWidget_CM1"] , None))
    self.label_True_CM1.setText(_translate("MainWindow", "T\n"
"R\n"
"U\n"
"E",None))
    self.pushButton_CM1_to_Clipboard.setText(_translate("MainWindow", "To Clipboard",None))
    self.label_Pred_CM1.setText(_translate("MainWindow", "PREDICTED",None))
    self.tableWidget_CM2.setToolTip(_translate("MainWindow",tooltips["tableWidget_CM2"] , None))

    self.label_True_CM2.setText(_translate("MainWindow", "T\n"
"R\n"
"U\n"
"E",None))
    self.pushButton_CM2_to_Clipboard.setText(_translate("MainWindow", "To Clipboard",None))
    self.label_Pred_CM2.setText(_translate("MainWindow", "PREDICTED",None))
    
    self.tableWidget_AccPrecSpec.setToolTip(_translate("MainWindow",tooltips["tableWidget_AccPrecSpec"] , None))
    self.groupBox_probHistPlot.setTitle(_translate("MainWindow", "Probability histogram", None))
    self.groupBox_probHistPlot.setToolTip(_translate("MainWindow",tooltips["groupBox_probHistPlot"] , None))

    self.groupBox_3rdPlot.setTitle(_translate("MainWindow", "3rd plot", None))
    self.tabWidget_Modelbuilder.setTabText(self.tabWidget_Modelbuilder.indexOf(self.tab_AssessModel), _translate("MainWindow", "Assess Model", None))


    #Plotting Peakdet-Tab
    self.groupBox_plottingregion.setTitle(_translate("MainWindow", "Plotting region", None))
    self.groupBox_plottingOptions.setTitle(_translate("MainWindow", "Plotting options", None))
    self.checkBox_fl1.setText(_translate("MainWindow", "FL1", None))
    self.checkBox_fl2.setText(_translate("MainWindow", "FL2", None))
    self.checkBox_fl3.setText(_translate("MainWindow", "FL3", None))
    self.checkBox_centroid.setText(_translate("MainWindow", "Centroid", None))
    self.label_coloring.setText(_translate("MainWindow", "Image channel", None))
    self.checkBox_colorLog.setText(_translate("MainWindow", "Logscaled", None))
    self.pushButton_updateScatterPlot.setText(_translate("MainWindow", "Update", None))
    self.groupBox.setTitle(_translate("MainWindow", "Info", None))
    self.tabWidget_filter_peakdet.setTabText(self.tabWidget_filter_peakdet.indexOf(self.tab_filter), _translate("MainWindow", "Placeholder", None))
    self.groupBox_showCell.setTitle(_translate("MainWindow", "Show cell", None))
    self.groupBox_showSelectedPeaks.setTitle(_translate("MainWindow", "Select peaks manually", None))
    self.label_automatic.setText(_translate("MainWindow", "Automatic", None))
    self.pushButton_highestXPercent.setText(_translate("MainWindow", "Highest x %", None))
    self.label_remove.setText(_translate("MainWindow", "Remove", None))
    self.pushButton_selectPeakPos.setText(_translate("MainWindow", "Peak", None))
    self.pushButton_selectPeakRange.setText(_translate("MainWindow", "Range", None))
    self.pushButton_removeSelectedPeaks.setText(_translate("MainWindow", "Selected", None))
    self.pushButton_removeAllPeaks.setText(_translate("MainWindow", "All", None))
    self.groupBox_peakDetModel.setTitle(_translate("MainWindow", "Peak detection Model", None))
    self.pushButton_fitPeakDetModel.setText(_translate("MainWindow", "Fit model to peaks", None))
    self.pushButton_SavePeakDetModel.setText(_translate("MainWindow", "Save model", None))
    self.pushButton_loadPeakDetModel.setText(_translate("MainWindow", "Load model", None))
    self.radioButton_exportSelected.setText(_translate("MainWindow", "only selected", None))
    self.radioButton_exportAll.setText(_translate("MainWindow", "all", None))
    self.pushButton_export.setText(_translate("MainWindow", "Export to...", None))
    self.tabWidget_filter_peakdet.setTabText(self.tabWidget_filter_peakdet.indexOf(self.tab_peakdet), _translate("MainWindow", "Peakdetection", None))
    self.tabWidget_filter_peakdet.setTabText(self.tabWidget_filter_peakdet.indexOf(self.tab_defineModel), _translate("MainWindow", "Placeholder", None))
    self.tabWidget_Modelbuilder.setTabText(self.tabWidget_Modelbuilder.indexOf(self.tab_Plotting), _translate("MainWindow", "Plot/Peak", None))

    ##############################Python Tab###############################
    self.groupBox_pythonMenu.setTitle(_translate("MainWindow", "File", None))
    self.label_pythonCurrentFile.setText(_translate("MainWindow", "Current file:", None))
    self.groupBox_pythonEditor.setTitle(_translate("MainWindow", "Editor", None))
    self.pushButton_pythonInOpen.setText(_translate("MainWindow", "Open file..", None))
    self.pushButton_pythonSaveAs.setText(_translate("MainWindow", "Save as...", None))
    self.pushButton_pythonInClear.setText(_translate("MainWindow", "Clear", None))
    self.pushButton_pythonInRun.setText(_translate("MainWindow", "Run", None))
    self.groupBox_pythonConsole.setTitle(_translate("MainWindow", "Console", None))
    self.pushButton_pythonOutClear.setText(_translate("MainWindow", "Clear", None))
    self.pushButton_pythonOutRun.setText(_translate("MainWindow", "Run", None))
    self.tabWidget_Modelbuilder.setTabText(self.tabWidget_Modelbuilder.indexOf(self.tab_python), _translate("MainWindow", "Python", None))

    self.comboBox_chooseRtdcFile.setToolTip(_translate("MainWindow",tooltips["comboBox_chooseRtdcFile"] , None))
    self.comboBox_featurey.setToolTip(_translate("MainWindow",tooltips["comboBox_featurey"]  , None))
    self.comboBox_featurex.setToolTip(_translate("MainWindow",tooltips["comboBox_featurey"]  , None))
    self.widget_histx.setToolTip(_translate("MainWindow",tooltips["widget_histx"] , None))
    self.widget_histy.setToolTip(_translate("MainWindow", tooltips["widget_histy"], None))
    self.horizontalSlider_cellInd.setToolTip(_translate("MainWindow", tooltips["horizontalSlider_cellInd"] , None))
    self.spinBox_cellInd.setToolTip(_translate("MainWindow",tooltips["spinBox_cellInd"] , None))
    self.widget_scatter.setToolTip(_translate("MainWindow",tooltips["widget_scatter"] , None))
    self.checkBox_fl1.setToolTip(_translate("MainWindow",tooltips["checkBox_fl1"], None))
    self.checkBox_fl2.setToolTip(_translate("MainWindow", tooltips["checkBox_fl2"], None))
    self.checkBox_fl3.setToolTip(_translate("MainWindow", tooltips["checkBox_fl3"], None))
    self.checkBox_centroid.setToolTip(_translate("MainWindow",tooltips["checkBox_centroid"] , None))
    self.pushButton_selectPeakPos.setToolTip(_translate("MainWindow",tooltips["pushButton_selectPeakPos"] , None))
    self.pushButton_selectPeakRange.setToolTip(_translate("MainWindow",tooltips["pushButton_selectPeakRange"]  , None))
    self.pushButton_highestXPercent.setToolTip(_translate("MainWindow",tooltips["pushButton_highestXPercent"] , None))
    self.doubleSpinBox_highestXPercent.setToolTip(_translate("MainWindow",tooltips["pushButton_highestXPercent"]  , None))
    self.pushButton_removeSelectedPeaks.setToolTip(_translate("MainWindow",tooltips["pushButton_removeSelectedPeaks"] , None))
    self.pushButton_removeAllPeaks.setToolTip(_translate("MainWindow",tooltips["pushButton_removeAllPeaks"] , None))
    self.widget_showSelectedPeaks.setToolTip(_translate("MainWindow", tooltips["widget_showSelectedPeaks"], None))
    self.tableWidget_showSelectedPeaks.setToolTip(_translate("MainWindow",tooltips["tableWidget_showSelectedPeaks"] , None))
    self.groupBox_showCell.setToolTip(_translate("MainWindow", tooltips["groupBox_showCell"] , None))
    self.pushButton_updateScatterPlot.setToolTip(_translate("MainWindow",tooltips["pushButton_updateScatterPlot"] , None))
    self.tableWidget_peakModelParameters.setToolTip(_translate("MainWindow",tooltips["tableWidget_peakModelParameters"] , None))

    self.comboBox_peakDetModel.setToolTip(_translate("MainWindow",tooltips["tableWidget_peakModelParameters"] , None))
    self.pushButton_fitPeakDetModel.setToolTip(_translate("MainWindow",tooltips["pushButton_fitPeakDetModel"] , None))
    self.pushButton_SavePeakDetModel.setToolTip(_translate("MainWindow",tooltips["pushButton_SavePeakDetModel"] , None))
    self.pushButton_loadPeakDetModel.setToolTip(_translate("MainWindow",tooltips["pushButton_loadPeakDetModel"] , None))
    self.radioButton_exportSelected.setToolTip(_translate("MainWindow",tooltips["radioButton_exportSelected"] , None))
    self.radioButton_exportAll.setToolTip(_translate("MainWindow",tooltips["radioButton_exportAll"] , None))


    self.menuFile.setTitle(_translate("MainWindow", "File", None))
#    self.menuEdit.setTitle(_translate("MainWindow", "Edit", None))
    self.menu_Options.setTitle(_translate("MainWindow", "Options",None))
    self.menuLayout.setTitle(_translate("MainWindow", "Layout",None))
    self.menuExport.setTitle(_translate("MainWindow", "Export",None))
    self.menuGPU_Options.setTitle(_translate("MainWindow", "Multi-GPU",None))

    self.menu_Options.setToolTip(_translate("MainWindow", "<html><head/><body><p>menu_Options tooltip</p></body></html>", None))

    self.menu_Help.setTitle(_translate("MainWindow", "Help",None))
    self.actionDocumentation.setText(_translate("MainWindow", "Documentation",None))
    self.actionTerminology.setText(_translate("MainWindow", "Terminology",None))

    self.actionSoftware.setText(_translate("MainWindow", "Software",None))
    self.actionAbout.setText(_translate("MainWindow", "About",None))
    self.actionUpdate.setText(_translate("MainWindow", "Check for updates...",None))


    self.actionLoadSession.setText(_translate("MainWindow", "Load Session", None))
    self.actionSaveSession.setText(_translate("MainWindow", "Save Session", None))
    self.actionQuit.setText(_translate("MainWindow", "Quit", None))
    self.actionDataToRam.setText(_translate("MainWindow", "Data to RAM upon Initialization of Model", None))
    self.actionKeep_Data_in_RAM.setText(_translate("MainWindow", "Keep Data in RAM for multiple training sessions", None))
    self.actionVerbose.setText(_translate("MainWindow", "Verbose", None))
#        self.actionShowDataOverview.setText(_translate("MainWindow", "Show Data Overview", None))
    self.actionClearList.setText(_translate("MainWindow", "Clear List", None))
    self.actionDataToRamNow.setText(_translate("MainWindow", "Data to RAM now", None))
    self.actionRemoveSelected.setText(_translate("MainWindow", "Remove selected", None))
    self.actionSaveToPng.setText(_translate("MainWindow", "Export selected to .png/.jpg", None))
    self.actionClearMemory.setText(_translate("MainWindow", "Clear memory (CPU/GPU)", None))
    self.actionOpenTemp.setText(_translate("MainWindow", "Open temp directory", None))
    
    self.actionExport_Off.setText(_translate("MainWindow", "No exporting",None))
    self.actionExport_Original.setText(_translate("MainWindow", "Export Original Images",None))
    self.actionExport_Cropped.setText(_translate("MainWindow", "Export Cropped Images",None))
    self.actionLayout_Normal.setText(_translate("MainWindow", "Normal layout",None))
    self.actionLayout_Dark.setText(_translate("MainWindow", "Dark layout",None))
    self.actionLayout_DarkOrange.setText(_translate("MainWindow", "DarkOrange layout",None))
    self.actionIconTheme_1.setText(_translate("MainWindow", "Icon theme 1",None))
    self.actionIconTheme_2.setText(_translate("MainWindow", "Icon theme 2",None))

    self.actioncpu_merge.setText(_translate("MainWindow", "cpu_merge",None))
    self.actioncpu_merge.setToolTip(_translate("MainWindow", tooltips["actioncpu_merge"],None))
    self.actioncpu_relocation.setText(_translate("MainWindow", "cpu_relocation",None))
    self.actioncpu_relocation.setToolTip(_translate("MainWindow", tooltips["actioncpu_relocation"],None))
    self.actioncpu_weightmerge.setText(_translate("MainWindow", "cpu_weight_merge",None))
    self.actioncpu_weightmerge.setToolTip(_translate("MainWindow", tooltips["actioncpu_weightmerge"],None))

    self.actionTooltipOnOff.setText(_translate("MainWindow", "Show tooltips",None))

    #count the number of files in the temp. folder
    nr_temp_files = aid_bin.count_temp_folder()
    if nr_temp_files>0:
        #inform user
        self.statusbar.showMessage("Files are left in temporary folder. Find temp via: ->Options->Open temp directory",10000)    





##############################Function for Main UI End#############################







































class Fitting_Ui(QtWidgets.QWidget):
    def setupUi(self, Form):
        self.Form = Form
        Form.setObjectName(_fromUtf8("Form"))
        Form.resize(797, 786)

        self.gridLayout_slider_pop = QtWidgets.QGridLayout(Form)
        self.gridLayout_slider_pop.setObjectName("gridLayout_slider_pop")
        self.verticalLayout_4_pop = QtWidgets.QVBoxLayout()
        self.verticalLayout_4_pop.setObjectName("verticalLayout_4_pop")
        self.horizontalLayout_pop = QtWidgets.QHBoxLayout()
        self.horizontalLayout_pop.setObjectName("horizontalLayout_pop")
        self.tableWidget_HistoryInfo_pop = QtWidgets.QTableWidget(Form)
        self.tableWidget_HistoryInfo_pop.setObjectName("tableWidget_HistoryInfo_pop")
        self.tableWidget_HistoryInfo_pop.setColumnCount(0)
        self.tableWidget_HistoryInfo_pop.setRowCount(0)
        self.horizontalLayout_pop.addWidget(self.tableWidget_HistoryInfo_pop)
        self.verticalLayout_2_pop = QtWidgets.QVBoxLayout()
        self.verticalLayout_2_pop.setObjectName("verticalLayout_2_pop")
        self.pushButton_UpdatePlot_pop = QtWidgets.QPushButton(Form)
        self.pushButton_UpdatePlot_pop.setObjectName("pushButton_UpdatePlot_pop")
        self.verticalLayout_2_pop.addWidget(self.pushButton_UpdatePlot_pop)

        self.checkBox_realTimePlotting_pop = QtWidgets.QCheckBox(Form)
        self.checkBox_realTimePlotting_pop.setObjectName("checkBox_realTimePlotting_pop")
        self.verticalLayout_2_pop.addWidget(self.checkBox_realTimePlotting_pop)
        self.horizontalLayout_rtepochs_pop = QtWidgets.QHBoxLayout()
        self.label_realTimeEpochs_pop = QtWidgets.QLabel(Form)
        self.label_realTimeEpochs_pop.setObjectName("label_realTimeEpochs_pop")
        self.horizontalLayout_rtepochs_pop.addWidget(self.label_realTimeEpochs_pop)
        self.spinBox_realTimeEpochs = QtWidgets.QSpinBox(Form)
        self.spinBox_realTimeEpochs.setObjectName("spinBox_realTimeEpochs")
        self.horizontalLayout_rtepochs_pop.addWidget(self.spinBox_realTimeEpochs)        
        self.verticalLayout_2_pop.addLayout(self.horizontalLayout_rtepochs_pop)
        
        self.horizontalLayout_pop.addLayout(self.verticalLayout_2_pop)
        self.verticalLayout_4_pop.addLayout(self.horizontalLayout_pop)
        self.verticalLayout_3_pop = QtWidgets.QVBoxLayout()
        self.verticalLayout_3_pop.setObjectName("verticalLayout_3_pop")
        self.widget_pop = pg.GraphicsLayoutWidget(Form)#QtWidgets.QWidget(Form)
        self.widget_pop.setMinimumSize(QtCore.QSize(771, 331))
        self.widget_pop.setObjectName("widget_pop")
        self.verticalLayout_3_pop.addWidget(self.widget_pop)
        self.splitter_pop = QtWidgets.QSplitter(Form)
        self.splitter_pop.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_pop.setObjectName("splitter_pop")
        self.groupBox_FittingInfo_pop = QtWidgets.QGroupBox(self.splitter_pop)
        self.groupBox_FittingInfo_pop.setObjectName("groupBox_FittingInfo_pop")
        self.gridLayout_2_pop = QtWidgets.QGridLayout(self.groupBox_FittingInfo_pop)
        self.gridLayout_2_pop.setObjectName("gridLayout_2_pop")
        self.progressBar_Fitting_pop = QtWidgets.QProgressBar(self.groupBox_FittingInfo_pop)
        self.progressBar_Fitting_pop.setProperty("value", 24)
        self.progressBar_Fitting_pop.setObjectName("progressBar_Fitting_pop")
        self.gridLayout_2_pop.addWidget(self.progressBar_Fitting_pop, 0, 0, 1, 1)
        self.textBrowser_FittingInfo = QtWidgets.QTextBrowser(self.groupBox_FittingInfo_pop)
        self.textBrowser_FittingInfo.setObjectName("textBrowser_FittingInfo")
        self.gridLayout_2_pop.addWidget(self.textBrowser_FittingInfo, 1, 0, 1, 1)
        self.horizontalLayout_saveClearText_pop = QtWidgets.QHBoxLayout()
        self.horizontalLayout_saveClearText_pop.setObjectName("horizontalLayout_saveClearText_pop")
        self.pushButton_saveTextWindow_pop = QtWidgets.QPushButton(self.groupBox_FittingInfo_pop)
        self.pushButton_saveTextWindow_pop.setObjectName("pushButton_saveTextWindow_pop")
        self.horizontalLayout_saveClearText_pop.addWidget(self.pushButton_saveTextWindow_pop)
        self.pushButton_clearTextWindow_pop = QtWidgets.QPushButton(self.groupBox_FittingInfo_pop)
        self.pushButton_clearTextWindow_pop.setObjectName("pushButton_clearTextWindow_pop")
        self.horizontalLayout_saveClearText_pop.addWidget(self.pushButton_clearTextWindow_pop)
        self.gridLayout_2_pop.addLayout(self.horizontalLayout_saveClearText_pop, 2, 0, 1, 1)
        self.groupBox_ChangeModel_pop = QtWidgets.QGroupBox(self.splitter_pop)
        self.groupBox_ChangeModel_pop.setEnabled(True)
        self.groupBox_ChangeModel_pop.setCheckable(False)
        self.groupBox_ChangeModel_pop.setObjectName("groupBox_ChangeModel_pop")
        self.gridLayout_3_pop = QtWidgets.QGridLayout(self.groupBox_ChangeModel_pop)
        self.gridLayout_3_pop.setObjectName("gridLayout_3_pop")
        self.tabWidget_DefineModel_pop = QtWidgets.QTabWidget(self.groupBox_ChangeModel_pop)
        self.tabWidget_DefineModel_pop.setEnabled(True)
        self.tabWidget_DefineModel_pop.setUsesScrollButtons(True)
        self.tabWidget_DefineModel_pop.setObjectName("tabWidget_DefineModel_pop")
        
        self.tab_DefineModel_pop = QtWidgets.QWidget()
        self.tab_DefineModel_pop.setObjectName("tab_DefineModel_pop")
        self.gridLayout = QtWidgets.QGridLayout(self.tab_DefineModel_pop)
        self.gridLayout.setObjectName("gridLayout")
        self.scrollArea_defineModel_pop = QtWidgets.QScrollArea(self.tab_DefineModel_pop)
        self.scrollArea_defineModel_pop.setWidgetResizable(True)
        self.scrollArea_defineModel_pop.setObjectName("scrollArea_defineModel_pop")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 523, 352))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.gridLayout_13 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents)
        self.gridLayout_13.setObjectName("gridLayout_13")
        self.verticalLayout_defineModel_pop = QtWidgets.QVBoxLayout()
        self.verticalLayout_defineModel_pop.setObjectName("verticalLayout_defineModel_pop")
        self.horizontalLayout_2_pop = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2_pop.setObjectName("horizontalLayout_2_pop")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2_pop.addItem(spacerItem)
        self.label_ModelGeomIcon_pop = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_ModelGeomIcon_pop.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_ModelGeomIcon_pop.setText("")
        self.label_ModelGeomIcon_pop.setObjectName("label_ModelGeomIcon_pop")
        self.horizontalLayout_2_pop.addWidget(self.label_ModelGeomIcon_pop)
        self.label_ModelGeom_pop = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_ModelGeom_pop.setObjectName("label_ModelGeom_pop")
        self.label_ModelGeom_pop.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.horizontalLayout_2_pop.addWidget(self.label_ModelGeom_pop)
        self.comboBox_ModelSelection_pop = QtWidgets.QComboBox(self.scrollAreaWidgetContents)
        self.comboBox_ModelSelection_pop.setEnabled(False)
        self.comboBox_ModelSelection_pop.setMinimumSize(QtCore.QSize(200, 20))
        self.comboBox_ModelSelection_pop.setMaximumSize(QtCore.QSize(16777215, 20))
        self.comboBox_ModelSelection_pop.setEditable(True)
        self.comboBox_ModelSelection_pop.setObjectName("comboBox_ModelSelection_pop")
        self.horizontalLayout_2_pop.addWidget(self.comboBox_ModelSelection_pop)
        self.verticalLayout_defineModel_pop.addLayout(self.horizontalLayout_2_pop)
        self.groupBox_expt_imgProc_pop = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
        self.groupBox_expt_imgProc_pop.setObjectName("groupBox_expt_imgProc_pop")
        self.gridLayout_10 = QtWidgets.QGridLayout(self.groupBox_expt_imgProc_pop)
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.horizontalLayout_8_pop = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8_pop.setObjectName("horizontalLayout_8_pop")
        self.label_CropIcon_pop = QtWidgets.QLabel(self.groupBox_expt_imgProc_pop)
        self.label_CropIcon_pop.setText("")
        #self.label_CropIcon_pop.setPixmap(QtGui.QPixmap("../../51 GUI_MORE-ACS_Modelbuilder/MORE-ModelMaker_v0.1.6_ForPython_3.5/art/cropping_small.png"))
        self.label_CropIcon_pop.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_CropIcon_pop.setObjectName("label_CropIcon_pop")
        self.horizontalLayout_8_pop.addWidget(self.label_CropIcon_pop)
        self.label_Crop_pop = QtWidgets.QLabel(self.groupBox_expt_imgProc_pop)
        self.label_Crop_pop.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_Crop_pop.setObjectName("label_Crop_pop")
        self.horizontalLayout_8_pop.addWidget(self.label_Crop_pop)
        self.spinBox_imagecrop_pop = QtWidgets.QSpinBox(self.groupBox_expt_imgProc_pop)
        self.spinBox_imagecrop_pop.setEnabled(False)
        self.spinBox_imagecrop_pop.setMaximum(9999)
        self.spinBox_imagecrop_pop.setObjectName("spinBox_imagecrop_pop")
        self.horizontalLayout_8_pop.addWidget(self.spinBox_imagecrop_pop)
        self.gridLayout_10.addLayout(self.horizontalLayout_8_pop, 0, 0, 1, 1)
        self.horizontalLayout_5_pop = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5_pop.setObjectName("horizontalLayout_5_pop")
        self.label_NormalizationIcon_pop = QtWidgets.QLabel(self.groupBox_expt_imgProc_pop)
        self.label_NormalizationIcon_pop.setText("")
        #self.label_NormalizationIcon_pop.setPixmap(QtGui.QPixmap("../013_AIDeveloper_0.0.8_dev1/art/Icon theme 1/normalization.png"))
        self.label_NormalizationIcon_pop.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_NormalizationIcon_pop.setObjectName("label_NormalizationIcon_pop")
        self.horizontalLayout_5_pop.addWidget(self.label_NormalizationIcon_pop)
        self.label_Normalization_pop = QtWidgets.QLabel(self.groupBox_expt_imgProc_pop)
        self.label_Normalization_pop.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.label_Normalization_pop.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_Normalization_pop.setObjectName("label_Normalization_pop")
        self.horizontalLayout_5_pop.addWidget(self.label_Normalization_pop)
        self.comboBox_Normalization_pop = QtWidgets.QComboBox(self.groupBox_expt_imgProc_pop)
        self.comboBox_Normalization_pop.setEnabled(False)
        self.comboBox_Normalization_pop.setMinimumSize(QtCore.QSize(100, 0))
        self.comboBox_Normalization_pop.setObjectName("comboBox_Normalization_pop")
        self.horizontalLayout_5_pop.addWidget(self.comboBox_Normalization_pop)
        self.gridLayout_10.addLayout(self.horizontalLayout_5_pop, 0, 1, 1, 1)

        self.gridLayout_10.addLayout(self.horizontalLayout_5_pop, 0, 1, 1, 1)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_zoomIcon = QtWidgets.QLabel(self.groupBox_expt_imgProc_pop)
        self.label_zoomIcon.setText("")
        #self.label_zoomIcon.setPixmap(QtGui.QPixmap("../000_Icons/Version_2/zoom_order.png"))
        self.label_zoomIcon.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_zoomIcon.setObjectName("label_zoomIcon")
        self.horizontalLayout_6.addWidget(self.label_zoomIcon)
        self.label_zoomOrder = QtWidgets.QLabel(self.groupBox_expt_imgProc_pop)
        self.label_zoomOrder.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_zoomOrder.setObjectName("label_zoomOrder")
        self.horizontalLayout_6.addWidget(self.label_zoomOrder)
        self.comboBox_zoomOrder = QtWidgets.QComboBox(self.groupBox_expt_imgProc_pop)
        self.comboBox_zoomOrder.setEnabled(False)
        self.comboBox_zoomOrder.setMaximumSize(QtCore.QSize(100, 16777215))
        self.comboBox_zoomOrder.setObjectName("comboBox_zoomOrder")
        self.comboBox_zoomOrder.addItem("")
        self.comboBox_zoomOrder.addItem("")
        self.comboBox_zoomOrder.addItem("")
        self.comboBox_zoomOrder.addItem("")
        self.comboBox_zoomOrder.addItem("")
        self.comboBox_zoomOrder.setMaximumSize(QtCore.QSize(100, 16777215))

        self.horizontalLayout_6.addWidget(self.comboBox_zoomOrder)
        self.gridLayout_10.addLayout(self.horizontalLayout_6, 2, 0, 1, 1)


        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_padIcon_pop = QtWidgets.QLabel(self.groupBox_expt_imgProc_pop)
        self.label_padIcon_pop.setText("")
        #self.label_padIcon_pop.setPixmap(QtGui.QPixmap("../013_AIDeveloper_0.0.8_dev1/art/Icon theme 1/padding.png"))
        self.label_padIcon_pop.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_padIcon_pop.setObjectName("label_padIcon_pop")
        self.horizontalLayout_2.addWidget(self.label_padIcon_pop)
        self.label_paddingMode_pop = QtWidgets.QLabel(self.groupBox_expt_imgProc_pop)
        self.label_paddingMode_pop.setObjectName("label_paddingMode_pop")
        self.horizontalLayout_2.addWidget(self.label_paddingMode_pop)
        self.comboBox_paddingMode_pop = QtWidgets.QComboBox(self.groupBox_expt_imgProc_pop)
        self.comboBox_paddingMode_pop.setEnabled(True)
        self.comboBox_paddingMode_pop.setObjectName("comboBox_paddingMode_pop")
        self.comboBox_paddingMode_pop.addItem("")
        self.comboBox_paddingMode_pop.addItem("")
        self.comboBox_paddingMode_pop.addItem("")
        self.comboBox_paddingMode_pop.addItem("")
        self.comboBox_paddingMode_pop.addItem("")
        self.comboBox_paddingMode_pop.addItem("")
        self.comboBox_paddingMode_pop.addItem("")

        self.horizontalLayout_2.addWidget(self.comboBox_paddingMode_pop)
        self.gridLayout_10.addLayout(self.horizontalLayout_2, 1, 0, 1, 1)
        self.horizontalLayout_3_pop_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3_pop_2.setObjectName("horizontalLayout_3_pop_2")
        self.label_colorModeIcon_pop = QtWidgets.QLabel(self.groupBox_expt_imgProc_pop)
        self.label_colorModeIcon_pop.setText("")
        #self.label_colorModeIcon_pop.setPixmap(QtGui.QPixmap("../../51 GUI_MORE-ACS_Modelbuilder/Icons/color_mode.png"))
        self.label_colorModeIcon_pop.setScaledContents(False)
        self.label_colorModeIcon_pop.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_colorModeIcon_pop.setObjectName("label_colorModeIcon_pop")
        self.horizontalLayout_3_pop_2.addWidget(self.label_colorModeIcon_pop)
        self.label_colorMode_pop = QtWidgets.QLabel(self.groupBox_expt_imgProc_pop)
        self.label_colorMode_pop.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_colorMode_pop.setObjectName("label_colorMode_pop")
        self.horizontalLayout_3_pop_2.addWidget(self.label_colorMode_pop)
        self.comboBox_colorMode_pop = QtWidgets.QComboBox(self.groupBox_expt_imgProc_pop)
        self.comboBox_colorMode_pop.setEnabled(False)
        self.comboBox_colorMode_pop.setObjectName("comboBox_colorMode_pop")
        self.horizontalLayout_3_pop_2.addWidget(self.comboBox_colorMode_pop)
        self.gridLayout_10.addLayout(self.horizontalLayout_3_pop_2, 1, 1, 1, 1)
        self.verticalLayout_defineModel_pop.addWidget(self.groupBox_expt_imgProc_pop)
        self.horizontalLayout_6_pop = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6_pop.setObjectName("horizontalLayout_6_pop")
        self.verticalLayout_defineModel_pop.addLayout(self.horizontalLayout_6_pop)
        self.groupBox_system_pop = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
        self.groupBox_system_pop.setObjectName("groupBox_system_pop")
        self.gridLayout_48 = QtWidgets.QGridLayout(self.groupBox_system_pop)
        self.gridLayout_48.setObjectName("gridLayout_48")
        self.radioButton_gpu_pop = QtWidgets.QRadioButton(self.groupBox_system_pop)
        self.radioButton_gpu_pop.setEnabled(False)
        self.radioButton_gpu_pop.setObjectName("radioButton_gpu_pop")
        self.gridLayout_48.addWidget(self.radioButton_gpu_pop, 1, 3, 1, 1)
        self.comboBox_cpu_pop = QtWidgets.QComboBox(self.groupBox_system_pop)
        self.comboBox_cpu_pop.setObjectName("comboBox_cpu_pop")
        self.comboBox_cpu_pop.setEnabled(False)
        self.gridLayout_48.addWidget(self.comboBox_cpu_pop, 0, 4, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(198, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_48.addItem(spacerItem, 0, 5, 1, 3)
        self.radioButton_cpu_pop = QtWidgets.QRadioButton(self.groupBox_system_pop)
        self.radioButton_cpu_pop.setEnabled(False)
        self.radioButton_cpu_pop.setObjectName("radioButton_cpu_pop")
        self.gridLayout_48.addWidget(self.radioButton_cpu_pop, 0, 3, 1, 1)
        self.doubleSpinBox_memory_pop = QtWidgets.QDoubleSpinBox(self.groupBox_system_pop)
        self.doubleSpinBox_memory_pop.setEnabled(False)
        self.doubleSpinBox_memory_pop.setObjectName("doubleSpinBox_memory_pop")
        self.gridLayout_48.addWidget(self.doubleSpinBox_memory_pop, 1, 7, 1, 1)
        self.comboBox_gpu_pop = QtWidgets.QComboBox(self.groupBox_system_pop)
        self.comboBox_gpu_pop.setEnabled(False)
        self.comboBox_gpu_pop.setObjectName("comboBox_gpu_pop")
        self.gridLayout_48.addWidget(self.comboBox_gpu_pop, 1, 4, 1, 2)
        self.label_memory_pop = QtWidgets.QLabel(self.groupBox_system_pop)
        self.label_memory_pop.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_memory_pop.setObjectName("label_memory_pop")
        self.gridLayout_48.addWidget(self.label_memory_pop, 1, 6, 1, 1)
        self.line_nrEpochs_cpu_pop = QtWidgets.QFrame(self.groupBox_system_pop)
        self.line_nrEpochs_cpu_pop.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_nrEpochs_cpu_pop.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_nrEpochs_cpu_pop.setObjectName("line_nrEpochs_cpu_pop")
        self.gridLayout_48.addWidget(self.line_nrEpochs_cpu_pop, 0, 2, 2, 1)
        self.horizontalLayout_4_pop = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4_pop.setObjectName("horizontalLayout_4_pop")
        self.label_Crop_NrEpochsIcon_pop = QtWidgets.QLabel(self.groupBox_system_pop)
        self.label_Crop_NrEpochsIcon_pop.setText("")
        self.label_Crop_NrEpochsIcon_pop.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_Crop_NrEpochsIcon_pop.setObjectName("label_Crop_NrEpochsIcon_pop")
        self.horizontalLayout_4_pop.addWidget(self.label_Crop_NrEpochsIcon_pop)
        self.label_Crop_NrEpochs_pop = QtWidgets.QLabel(self.groupBox_system_pop)
        self.label_Crop_NrEpochs_pop.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_Crop_NrEpochs_pop.setObjectName("label_Crop_NrEpochs_pop")
        self.horizontalLayout_4_pop.addWidget(self.label_Crop_NrEpochs_pop)
        self.spinBox_NrEpochs = QtWidgets.QSpinBox(self.groupBox_system_pop)
        self.spinBox_NrEpochs.setMaximum(999999999)
        self.spinBox_NrEpochs.setObjectName("spinBox_NrEpochs")
        self.horizontalLayout_4_pop.addWidget(self.spinBox_NrEpochs)
        self.gridLayout_48.addLayout(self.horizontalLayout_4_pop, 0, 1, 1, 1)
        
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_saveMetaEvery = QtWidgets.QLabel(self.groupBox_system_pop)
        self.label_saveMetaEvery.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_saveMetaEvery.setObjectName("label_saveMetaEvery")
        self.horizontalLayout_4.addWidget(self.label_saveMetaEvery)
        self.spinBox_saveMetaEvery = QtWidgets.QSpinBox(self.groupBox_system_pop)
        self.spinBox_saveMetaEvery.setMinimum(1)
        self.spinBox_saveMetaEvery.setMaximum(999999)
        self.spinBox_saveMetaEvery.setProperty("value", 30)
        self.spinBox_saveMetaEvery.setObjectName("spinBox_saveMetaEvery")
        self.horizontalLayout_4.addWidget(self.spinBox_saveMetaEvery)
        self.gridLayout_48.addLayout(self.horizontalLayout_4, 1, 1, 1, 1)

        self.verticalLayout_defineModel_pop.addWidget(self.groupBox_system_pop)
        self.horizontalLayout_modelname_2_pop = QtWidgets.QHBoxLayout()
        self.horizontalLayout_modelname_2_pop.setObjectName("horizontalLayout_modelname_2_pop")
        self.pushButton_modelname_pop = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        self.pushButton_modelname_pop.setEnabled(False)
        self.pushButton_modelname_pop.setObjectName("pushButton_modelname_pop")
        self.horizontalLayout_modelname_2_pop.addWidget(self.pushButton_modelname_pop)
        self.lineEdit_modelname_pop = QtWidgets.QLineEdit(self.scrollAreaWidgetContents)
        self.lineEdit_modelname_pop.setEnabled(False)
        self.lineEdit_modelname_pop.setMinimumSize(QtCore.QSize(0, 22))
        self.lineEdit_modelname_pop.setMaximumSize(QtCore.QSize(16777215, 22))
        self.lineEdit_modelname_pop.setObjectName("lineEdit_modelname_pop")
        self.horizontalLayout_modelname_2_pop.addWidget(self.lineEdit_modelname_pop)
        self.verticalLayout_defineModel_pop.addLayout(self.horizontalLayout_modelname_2_pop)
        self.line2_pop = QtWidgets.QFrame(self.scrollAreaWidgetContents)
        self.line2_pop.setFrameShape(QtWidgets.QFrame.HLine)
        self.line2_pop.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line2_pop.setObjectName("line2_pop")
        self.verticalLayout_defineModel_pop.addWidget(self.line2_pop)
        self.horizontalLayout_9_pop = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9_pop.setObjectName("horizontalLayout_9_pop")
        self.pushButton_showModelSumm_pop = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        self.pushButton_showModelSumm_pop.setObjectName("pushButton_showModelSumm_pop")
        self.horizontalLayout_9_pop.addWidget(self.pushButton_showModelSumm_pop)
        self.pushButton_saveModelSumm_pop = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        self.pushButton_saveModelSumm_pop.setObjectName("pushButton_saveModelSumm_pop")
        self.horizontalLayout_9_pop.addWidget(self.pushButton_saveModelSumm_pop)
        self.verticalLayout_defineModel_pop.addLayout(self.horizontalLayout_9_pop)
        self.horizontalLayout_7_pop = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7_pop.setObjectName("horizontalLayout_7_pop")
        self.verticalLayout_defineModel_pop.addLayout(self.horizontalLayout_7_pop)
        self.gridLayout_13.addLayout(self.verticalLayout_defineModel_pop, 0, 0, 1, 1)
        self.scrollArea_defineModel_pop.setWidget(self.scrollAreaWidgetContents)
        self.gridLayout.addWidget(self.scrollArea_defineModel_pop, 0, 0, 1, 1)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.checkBox_ApplyNextEpoch = QtWidgets.QCheckBox(self.tab_DefineModel_pop)
        self.checkBox_ApplyNextEpoch.setAutoFillBackground(False)
        self.checkBox_ApplyNextEpoch.setTristate(False)
        self.checkBox_ApplyNextEpoch.setObjectName("checkBox_ApplyNextEpoch")
        self.horizontalLayout_3.addWidget(self.checkBox_ApplyNextEpoch)
        self.checkBox_saveEpoch_pop = QtWidgets.QCheckBox(self.tab_DefineModel_pop)
        self.checkBox_saveEpoch_pop.setObjectName("checkBox_saveEpoch_pop")
        self.horizontalLayout_3.addWidget(self.checkBox_saveEpoch_pop)
        self.pushButton_Pause_pop = QtWidgets.QPushButton(self.tab_DefineModel_pop)
        self.pushButton_Pause_pop.setMinimumSize(QtCore.QSize(121, 28))
        self.pushButton_Pause_pop.setMaximumSize(QtCore.QSize(121, 28))
        self.pushButton_Pause_pop.setObjectName("pushButton_Pause_pop")
        self.horizontalLayout_3.addWidget(self.pushButton_Pause_pop)
        self.pushButton_Stop_pop = QtWidgets.QPushButton(self.tab_DefineModel_pop)
        self.pushButton_Stop_pop.setMinimumSize(QtCore.QSize(41, 28))
        self.pushButton_Stop_pop.setMaximumSize(QtCore.QSize(93, 28))
        self.pushButton_Stop_pop.setObjectName("pushButton_Stop_pop")
        self.horizontalLayout_3.addWidget(self.pushButton_Stop_pop)
        self.gridLayout.addLayout(self.horizontalLayout_3, 1, 0, 1, 1)

        self.tabWidget_DefineModel_pop.addTab(self.tab_DefineModel_pop, "")
        self.tab_KerasImgAug_pop = QtWidgets.QWidget()
        self.tab_KerasImgAug_pop.setObjectName("tab_KerasImgAug_pop")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.tab_KerasImgAug_pop)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.verticalLayout_9_pop = QtWidgets.QVBoxLayout()
        self.verticalLayout_9_pop.setObjectName("verticalLayout_9_pop")
        self.horizontalLayout_10_pop = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10_pop.setObjectName("horizontalLayout_10_pop")
        self.label_RefreshAfterEpochs_pop = QtWidgets.QLabel(self.tab_KerasImgAug_pop)
        self.label_RefreshAfterEpochs_pop.setObjectName("label_RefreshAfterEpochs_pop")
        self.horizontalLayout_10_pop.addWidget(self.label_RefreshAfterEpochs_pop)
        self.spinBox_RefreshAfterEpochs_pop = QtWidgets.QSpinBox(self.tab_KerasImgAug_pop)
        self.spinBox_RefreshAfterEpochs_pop.setObjectName("spinBox_RefreshAfterEpochs_pop")
        self.horizontalLayout_10_pop.addWidget(self.spinBox_RefreshAfterEpochs_pop)
        self.verticalLayout_9_pop.addLayout(self.horizontalLayout_10_pop)
        self.verticalLayout_10_pop = QtWidgets.QVBoxLayout()
        self.verticalLayout_10_pop.setObjectName("verticalLayout_10_pop")
        self.horizontalLayout_11_pop = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11_pop.setObjectName("horizontalLayout_11_pop")
        self.checkBox_HorizFlip_pop = QtWidgets.QCheckBox(self.tab_KerasImgAug_pop)
        self.checkBox_HorizFlip_pop.setObjectName("checkBox_HorizFlip_pop")
        self.horizontalLayout_11_pop.addWidget(self.checkBox_HorizFlip_pop)
        self.checkBox_VertFlip_pop = QtWidgets.QCheckBox(self.tab_KerasImgAug_pop)
        self.checkBox_VertFlip_pop.setObjectName("checkBox_VertFlip_pop")
        self.horizontalLayout_11_pop.addWidget(self.checkBox_VertFlip_pop)
        self.verticalLayout_10_pop.addLayout(self.horizontalLayout_11_pop)
        self.splitter_2_pop = QtWidgets.QSplitter(self.tab_KerasImgAug_pop)
        self.splitter_2_pop.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_2_pop.setObjectName("splitter_2_pop")
        self.layoutWidget_4 = QtWidgets.QWidget(self.splitter_2_pop)
        self.layoutWidget_4.setObjectName("layoutWidget_4")
        self.verticalLayout_11_pop = QtWidgets.QVBoxLayout(self.layoutWidget_4)
        self.verticalLayout_11_pop.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_11_pop.setObjectName("verticalLayout_11_pop")
        self.label_Rotation_pop = QtWidgets.QCheckBox(self.layoutWidget_4)
        self.label_Rotation_pop.setObjectName("label_Rotation_pop")
        self.verticalLayout_11_pop.addWidget(self.label_Rotation_pop)
        self.label_width_shift_pop = QtWidgets.QCheckBox(self.layoutWidget_4)
        self.label_width_shift_pop.setObjectName("label_width_shift_pop")
        self.verticalLayout_11_pop.addWidget(self.label_width_shift_pop)
        self.label_height_shift_pop = QtWidgets.QCheckBox(self.layoutWidget_4)
        self.label_height_shift_pop.setObjectName("label_height_shift_pop")
        self.verticalLayout_11_pop.addWidget(self.label_height_shift_pop)
        self.label_zoom_pop = QtWidgets.QCheckBox(self.layoutWidget_4)
        self.label_zoom_pop.setObjectName("label_zoom_pop")
        self.verticalLayout_11_pop.addWidget(self.label_zoom_pop)
        self.label_shear_pop = QtWidgets.QCheckBox(self.layoutWidget_4)
        self.label_shear_pop.setObjectName("label_shear_pop")
        self.verticalLayout_11_pop.addWidget(self.label_shear_pop)
        self.layoutWidget_5 = QtWidgets.QWidget(self.splitter_2_pop)
        self.layoutWidget_5.setObjectName("layoutWidget_5")
        self.verticalLayout_12_pop = QtWidgets.QVBoxLayout(self.layoutWidget_5)
        self.verticalLayout_12_pop.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_12_pop.setObjectName("verticalLayout_12_pop")
        
        self.onlyFloat = QtGui.QDoubleValidator()        
        
        self.lineEdit_Rotation_pop = QtWidgets.QLineEdit(self.layoutWidget_5)
        self.lineEdit_Rotation_pop.setObjectName("lineEdit_Rotation_pop")
        self.lineEdit_Rotation_pop.setValidator(self.onlyFloat)
        self.verticalLayout_12_pop.addWidget(self.lineEdit_Rotation_pop)
        self.lineEdit_widthShift_pop = QtWidgets.QLineEdit(self.layoutWidget_5)
        self.lineEdit_widthShift_pop.setObjectName("lineEdit_widthShift_pop")
        self.lineEdit_widthShift_pop.setValidator(self.onlyFloat)

        self.verticalLayout_12_pop.addWidget(self.lineEdit_widthShift_pop)
        self.lineEdit_heightShift_pop = QtWidgets.QLineEdit(self.layoutWidget_5)
        self.lineEdit_heightShift_pop.setObjectName("lineEdit_heightShift_pop")
        self.lineEdit_heightShift_pop.setValidator(self.onlyFloat)

        self.verticalLayout_12_pop.addWidget(self.lineEdit_heightShift_pop)
        self.lineEdit_zoomRange_pop = QtWidgets.QLineEdit(self.layoutWidget_5)
        self.lineEdit_zoomRange_pop.setObjectName("lineEdit_zoomRange_pop")
        self.lineEdit_zoomRange_pop.setValidator(self.onlyFloat)

        self.verticalLayout_12_pop.addWidget(self.lineEdit_zoomRange_pop)
        self.lineEdit_shearRange_pop = QtWidgets.QLineEdit(self.layoutWidget_5)
        self.lineEdit_shearRange_pop.setObjectName("lineEdit_shearRange_pop")
        self.lineEdit_shearRange_pop.setValidator(self.onlyFloat)

        self.verticalLayout_12_pop.addWidget(self.lineEdit_shearRange_pop)
        self.verticalLayout_10_pop.addWidget(self.splitter_2_pop)
        self.verticalLayout_9_pop.addLayout(self.verticalLayout_10_pop)
        self.gridLayout_8.addLayout(self.verticalLayout_9_pop, 0, 0, 1, 1)
        self.tabWidget_DefineModel_pop.addTab(self.tab_KerasImgAug_pop, "")



        self.tab_BrightnessAug_pop = QtWidgets.QWidget()
        self.tab_BrightnessAug_pop.setObjectName("tab_BrightnessAug_pop")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.tab_BrightnessAug_pop)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.scrollArea_BrightnessAug_pop = QtWidgets.QScrollArea(self.tab_BrightnessAug_pop)
        self.scrollArea_BrightnessAug_pop.setWidgetResizable(True)
        self.scrollArea_BrightnessAug_pop.setObjectName("scrollArea_BrightnessAug_pop")
        self.scrollAreaWidgetContents_pop_2 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_pop_2.setGeometry(QtCore.QRect(0, 0, 423, 269))
        self.scrollAreaWidgetContents_pop_2.setObjectName("scrollAreaWidgetContents_pop_2")
        self.gridLayout_9 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents_pop_2)
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.label_RefreshAfterNrEpochs_pop = QtWidgets.QLabel(self.scrollAreaWidgetContents_pop_2)
        self.label_RefreshAfterNrEpochs_pop.setObjectName("label_RefreshAfterNrEpochs_pop")
        self.gridLayout_9.addWidget(self.label_RefreshAfterNrEpochs_pop, 0, 0, 1, 1)
        self.spinBox_RefreshAfterNrEpochs_pop = QtWidgets.QSpinBox(self.scrollAreaWidgetContents_pop_2)
        self.spinBox_RefreshAfterNrEpochs_pop.setObjectName("spinBox_RefreshAfterNrEpochs_pop")
        self.gridLayout_9.addWidget(self.spinBox_RefreshAfterNrEpochs_pop, 0, 1, 1, 1)
        self.groupBox_BrightnessAugmentation_pop = QtWidgets.QGroupBox(self.scrollAreaWidgetContents_pop_2)
        self.groupBox_BrightnessAugmentation_pop.setObjectName("groupBox_BrightnessAugmentation_pop")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.groupBox_BrightnessAugmentation_pop)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.label_Plus_pop = QtWidgets.QCheckBox(self.groupBox_BrightnessAugmentation_pop)
        self.label_Plus_pop.setObjectName("label_Plus_pop")
        self.gridLayout_5.addWidget(self.label_Plus_pop, 0, 0, 1, 1)
        self.spinBox_PlusLower_pop = QtWidgets.QSpinBox(self.groupBox_BrightnessAugmentation_pop)
        self.spinBox_PlusLower_pop.setObjectName("spinBox_PlusLower_pop")
        self.gridLayout_5.addWidget(self.spinBox_PlusLower_pop, 0, 1, 1, 1)
        self.spinBox_PlusUpper_pop = QtWidgets.QSpinBox(self.groupBox_BrightnessAugmentation_pop)
        self.spinBox_PlusUpper_pop.setObjectName("spinBox_PlusUpper_pop")
        self.gridLayout_5.addWidget(self.spinBox_PlusUpper_pop, 0, 2, 1, 1)
        self.label_Mult_pop = QtWidgets.QCheckBox(self.groupBox_BrightnessAugmentation_pop)
        self.label_Mult_pop.setObjectName("label_Mult_pop")
        self.gridLayout_5.addWidget(self.label_Mult_pop, 1, 0, 1, 1)
        self.doubleSpinBox_MultLower_pop = QtWidgets.QDoubleSpinBox(self.groupBox_BrightnessAugmentation_pop)
        self.doubleSpinBox_MultLower_pop.setObjectName("doubleSpinBox_MultLower_pop")
        self.gridLayout_5.addWidget(self.doubleSpinBox_MultLower_pop, 1, 1, 1, 1)
        self.doubleSpinBox_MultUpper_pop = QtWidgets.QDoubleSpinBox(self.groupBox_BrightnessAugmentation_pop)
        self.doubleSpinBox_MultUpper_pop.setObjectName("doubleSpinBox_MultUpper_pop")
        self.gridLayout_5.addWidget(self.doubleSpinBox_MultUpper_pop, 1, 2, 1, 1)
        self.gridLayout_9.addWidget(self.groupBox_BrightnessAugmentation_pop, 1, 0, 1, 1)
        self.groupBox_GaussianNoise_pop = QtWidgets.QGroupBox(self.scrollAreaWidgetContents_pop_2)
        self.groupBox_GaussianNoise_pop.setObjectName("groupBox_GaussianNoise_pop")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.groupBox_GaussianNoise_pop)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.label_GaussianNoiseMean_pop = QtWidgets.QCheckBox(self.groupBox_GaussianNoise_pop)
        self.label_GaussianNoiseMean_pop.setObjectName("label_GaussianNoiseMean_pop")
        self.gridLayout_6.addWidget(self.label_GaussianNoiseMean_pop, 0, 0, 1, 1)
        self.doubleSpinBox_GaussianNoiseMean_pop = QtWidgets.QDoubleSpinBox(self.groupBox_GaussianNoise_pop)
        self.doubleSpinBox_GaussianNoiseMean_pop.setObjectName("doubleSpinBox_GaussianNoiseMean_pop")
        self.gridLayout_6.addWidget(self.doubleSpinBox_GaussianNoiseMean_pop, 0, 1, 1, 1)
        self.label_GaussianNoiseScale_pop = QtWidgets.QCheckBox(self.groupBox_GaussianNoise_pop)
        self.label_GaussianNoiseScale_pop.setObjectName("label_GaussianNoiseScale_pop")
        self.gridLayout_6.addWidget(self.label_GaussianNoiseScale_pop, 1, 0, 1, 1)
        self.doubleSpinBox_GaussianNoiseScale_pop = QtWidgets.QDoubleSpinBox(self.groupBox_GaussianNoise_pop)
        self.doubleSpinBox_GaussianNoiseScale_pop.setObjectName("doubleSpinBox_GaussianNoiseScale_pop")
        self.gridLayout_6.addWidget(self.doubleSpinBox_GaussianNoiseScale_pop, 1, 1, 1, 1)
        self.gridLayout_9.addWidget(self.groupBox_GaussianNoise_pop, 1, 1, 1, 1)
        self.groupBox_colorAugmentation_pop = QtWidgets.QGroupBox(self.scrollAreaWidgetContents_pop_2)
        self.groupBox_colorAugmentation_pop.setCheckable(False)
        self.groupBox_colorAugmentation_pop.setObjectName("groupBox_colorAugmentation_pop")
        self.gridLayout_15 = QtWidgets.QGridLayout(self.groupBox_colorAugmentation_pop)
        self.gridLayout_15.setObjectName("gridLayout_15")
        self.doubleSpinBox_contrastLower_pop = QtWidgets.QDoubleSpinBox(self.groupBox_colorAugmentation_pop)
        self.doubleSpinBox_contrastLower_pop.setMaximum(99.9)
        self.doubleSpinBox_contrastLower_pop.setSingleStep(0.1)
        #self.doubleSpinBox_contrastLower_pop.setProperty("value", 0.7)
        self.doubleSpinBox_contrastLower_pop.setObjectName("doubleSpinBox_contrastLower_pop")
        self.gridLayout_15.addWidget(self.doubleSpinBox_contrastLower_pop, 0, 1, 1, 1)
        self.doubleSpinBox_saturationHigher_pop = QtWidgets.QDoubleSpinBox(self.groupBox_colorAugmentation_pop)
        self.doubleSpinBox_saturationHigher_pop.setMaximum(99.9)
        self.doubleSpinBox_saturationHigher_pop.setSingleStep(0.1)
        #self.doubleSpinBox_saturationHigher_pop.setProperty("value", 1.3)
        self.doubleSpinBox_saturationHigher_pop.setObjectName("doubleSpinBox_saturationHigher_pop")
        self.gridLayout_15.addWidget(self.doubleSpinBox_saturationHigher_pop, 1, 2, 1, 1)
        self.doubleSpinBox_contrastHigher_pop = QtWidgets.QDoubleSpinBox(self.groupBox_colorAugmentation_pop)
        self.doubleSpinBox_contrastHigher_pop.setMaximum(99.9)
        self.doubleSpinBox_contrastHigher_pop.setSingleStep(0.1)
        #self.doubleSpinBox_contrastHigher_pop.setProperty("value", 1.3)
        self.doubleSpinBox_contrastHigher_pop.setObjectName("doubleSpinBox_contrastHigher_pop")
        self.gridLayout_15.addWidget(self.doubleSpinBox_contrastHigher_pop, 0, 2, 1, 1)
        self.checkBox_contrast_pop = QtWidgets.QCheckBox(self.groupBox_colorAugmentation_pop)
        self.checkBox_contrast_pop.setCheckable(True)
        self.checkBox_contrast_pop.setChecked(True)
        self.checkBox_contrast_pop.setObjectName("checkBox_contrast_pop")
        self.gridLayout_15.addWidget(self.checkBox_contrast_pop, 0, 0, 1, 1)
        self.doubleSpinBox_saturationLower_pop = QtWidgets.QDoubleSpinBox(self.groupBox_colorAugmentation_pop)
        self.doubleSpinBox_saturationLower_pop.setMaximum(99.9)
        self.doubleSpinBox_saturationLower_pop.setSingleStep(0.1)
        #self.doubleSpinBox_saturationLower_pop.setProperty("value", 0.7)
        self.doubleSpinBox_saturationLower_pop.setObjectName("doubleSpinBox_saturationLower_pop")
        self.gridLayout_15.addWidget(self.doubleSpinBox_saturationLower_pop, 1, 1, 1, 1)
        self.doubleSpinBox_hueDelta_pop = QtWidgets.QDoubleSpinBox(self.groupBox_colorAugmentation_pop)
        self.doubleSpinBox_hueDelta_pop.setMaximum(0.5)
        self.doubleSpinBox_hueDelta_pop.setSingleStep(0.01)
        #self.doubleSpinBox_hueDelta_pop.setProperty("value", 0.08)
        self.doubleSpinBox_hueDelta_pop.setObjectName("doubleSpinBox_hueDelta_pop")
        self.gridLayout_15.addWidget(self.doubleSpinBox_hueDelta_pop, 2, 1, 1, 1)
        self.checkBox_saturation_pop = QtWidgets.QCheckBox(self.groupBox_colorAugmentation_pop)
        self.checkBox_saturation_pop.setObjectName("checkBox_saturation_pop")
        self.gridLayout_15.addWidget(self.checkBox_saturation_pop, 1, 0, 1, 1)
        self.checkBox_hue_pop = QtWidgets.QCheckBox(self.groupBox_colorAugmentation_pop)
        self.checkBox_hue_pop.setObjectName("checkBox_hue_pop")
        self.gridLayout_15.addWidget(self.checkBox_hue_pop, 2, 0, 1, 1)
        self.gridLayout_9.addWidget(self.groupBox_colorAugmentation_pop, 2, 0, 1, 1)
        
        
        self.groupBox_blurringAug_pop = QtWidgets.QGroupBox(self.scrollAreaWidgetContents_pop_2)
        self.groupBox_blurringAug_pop.setObjectName("groupBox_blurringAug_pop")
        self.gridLayout_45 = QtWidgets.QGridLayout(self.groupBox_blurringAug_pop)
        self.gridLayout_45.setObjectName("gridLayout_45")
        self.gridLayout_blur_pop = QtWidgets.QGridLayout()
        self.gridLayout_blur_pop.setObjectName("gridLayout_blur_pop")
        self.label_motionBlurKernel_pop = QtWidgets.QLabel(self.groupBox_blurringAug_pop)
        self.label_motionBlurKernel_pop.setMaximumSize(QtCore.QSize(31, 16777215))
        self.label_motionBlurKernel_pop.setObjectName("label_motionBlurKernel_pop")
        self.gridLayout_blur_pop.addWidget(self.label_motionBlurKernel_pop, 2, 1, 1, 1)
        self.lineEdit_motionBlurAngle_pop = QtWidgets.QLineEdit(self.groupBox_blurringAug_pop)
        self.lineEdit_motionBlurAngle_pop.setMaximumSize(QtCore.QSize(100, 16777215))
        self.lineEdit_motionBlurAngle_pop.setInputMask("")
        self.lineEdit_motionBlurAngle_pop.setObjectName("lineEdit_motionBlurAngle_pop")
        validator = QtGui.QRegExpValidator(QtCore.QRegExp("^[-+]?[0-9]\\d{0,3},(\\d{3})$"))
        self.lineEdit_motionBlurAngle_pop.setValidator(validator)
        self.gridLayout_blur_pop.addWidget(self.lineEdit_motionBlurAngle_pop, 2, 4, 1, 1)
        self.label_avgBlurMin_pop = QtWidgets.QLabel(self.groupBox_blurringAug_pop)
        self.label_avgBlurMin_pop.setMaximumSize(QtCore.QSize(31, 16777215))
        self.label_avgBlurMin_pop.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_avgBlurMin_pop.setObjectName("label_avgBlurMin_pop")
        self.gridLayout_blur_pop.addWidget(self.label_avgBlurMin_pop, 0, 1, 1, 1)
        self.spinBox_gaussBlurMax_pop = QtWidgets.QSpinBox(self.groupBox_blurringAug_pop)
        self.spinBox_gaussBlurMax_pop.setObjectName("spinBox_gaussBlurMax_pop")
        self.gridLayout_blur_pop.addWidget(self.spinBox_gaussBlurMax_pop, 1, 4, 1, 1)
        self.checkBox_motionBlur_pop = QtWidgets.QCheckBox(self.groupBox_blurringAug_pop)
        self.checkBox_motionBlur_pop.setMaximumSize(QtCore.QSize(100, 16777215))
        self.checkBox_motionBlur_pop.setObjectName("checkBox_motionBlur_pop")
        self.gridLayout_blur_pop.addWidget(self.checkBox_motionBlur_pop, 2, 0, 1, 1)
        self.spinBox_avgBlurMin_pop = QtWidgets.QSpinBox(self.groupBox_blurringAug_pop)
        self.spinBox_avgBlurMin_pop.setObjectName("spinBox_avgBlurMin_pop")
        self.gridLayout_blur_pop.addWidget(self.spinBox_avgBlurMin_pop, 0, 2, 1, 1)
        self.spinBox_gaussBlurMin_pop = QtWidgets.QSpinBox(self.groupBox_blurringAug_pop)
        self.spinBox_gaussBlurMin_pop.setObjectName("spinBox_gaussBlurMin_pop")
        self.gridLayout_blur_pop.addWidget(self.spinBox_gaussBlurMin_pop, 1, 2, 1, 1)
        self.label_motionBlurAngle_pop = QtWidgets.QLabel(self.groupBox_blurringAug_pop)
        self.label_motionBlurAngle_pop.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.label_motionBlurAngle_pop.setObjectName("label_motionBlurAngle_pop")
        self.gridLayout_blur_pop.addWidget(self.label_motionBlurAngle_pop, 2, 3, 1, 1)
        self.label_gaussBlurMin_pop = QtWidgets.QLabel(self.groupBox_blurringAug_pop)
        self.label_gaussBlurMin_pop.setMaximumSize(QtCore.QSize(31, 16777215))
        self.label_gaussBlurMin_pop.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_gaussBlurMin_pop.setObjectName("label_gaussBlurMin_pop")
        self.gridLayout_blur_pop.addWidget(self.label_gaussBlurMin_pop, 1, 1, 1, 1)
        self.checkBox_gaussBlur_pop = QtWidgets.QCheckBox(self.groupBox_blurringAug_pop)
        self.checkBox_gaussBlur_pop.setObjectName("checkBox_gaussBlur_pop")
        self.gridLayout_blur_pop.addWidget(self.checkBox_gaussBlur_pop, 1, 0, 1, 1)
        self.spinBox_avgBlurMax_pop = QtWidgets.QSpinBox(self.groupBox_blurringAug_pop)
        self.spinBox_avgBlurMax_pop.setObjectName("spinBox_avgBlurMax_pop")
        self.gridLayout_blur_pop.addWidget(self.spinBox_avgBlurMax_pop, 0, 4, 1, 1)
        self.label_gaussBlurMax_pop = QtWidgets.QLabel(self.groupBox_blurringAug_pop)
        self.label_gaussBlurMax_pop.setMaximumSize(QtCore.QSize(31, 16777215))
        self.label_gaussBlurMax_pop.setObjectName("label_gaussBlurMax_pop")
        self.gridLayout_blur_pop.addWidget(self.label_gaussBlurMax_pop, 1, 3, 1, 1)
        self.checkBox_avgBlur_pop = QtWidgets.QCheckBox(self.groupBox_blurringAug_pop)
        self.checkBox_avgBlur_pop.setObjectName("checkBox_avgBlur_pop")
        self.gridLayout_blur_pop.addWidget(self.checkBox_avgBlur_pop, 0, 0, 1, 1)
        self.label_avgBlurMax_pop = QtWidgets.QLabel(self.groupBox_blurringAug_pop)
        self.label_avgBlurMax_pop.setMaximumSize(QtCore.QSize(31, 16777215))
        self.label_avgBlurMax_pop.setObjectName("label_avgBlurMax_pop")
        self.gridLayout_blur_pop.addWidget(self.label_avgBlurMax_pop, 0, 3, 1, 1)
        self.lineEdit_motionBlurKernel_pop = QtWidgets.QLineEdit(self.groupBox_blurringAug_pop)
        self.lineEdit_motionBlurKernel_pop.setMaximumSize(QtCore.QSize(100, 16777215))
        self.lineEdit_motionBlurKernel_pop.setInputMask("")
        self.lineEdit_motionBlurKernel_pop.setMaxLength(32767)
        self.lineEdit_motionBlurKernel_pop.setClearButtonEnabled(False)
        validator = QtGui.QRegExpValidator(QtCore.QRegExp("^\\d{1,3},(\\d{3})$"))
        self.lineEdit_motionBlurKernel_pop.setValidator(validator)
        self.lineEdit_motionBlurKernel_pop.setObjectName("lineEdit_motionBlurKernel_pop")
        self.gridLayout_blur_pop.addWidget(self.lineEdit_motionBlurKernel_pop, 2, 2, 1, 1)
        self.gridLayout_45.addLayout(self.gridLayout_blur_pop, 0, 0, 1, 1)
        self.gridLayout_9.addWidget(self.groupBox_blurringAug_pop, 2, 1, 1, 1)
        
        self.scrollArea_BrightnessAug_pop.setWidget(self.scrollAreaWidgetContents_pop_2)
        self.gridLayout_7.addWidget(self.scrollArea_BrightnessAug_pop, 0, 0, 1, 1)
        self.tabWidget_DefineModel_pop.addTab(self.tab_BrightnessAug_pop, "")
        self.tabWidget_DefineModel_pop.addTab(self.tab_BrightnessAug_pop, "")
        
        self.tab_ExampleImgs_pop = QtWidgets.QWidget()
        self.tab_ExampleImgs_pop.setObjectName("tab_ExampleImgs_pop")
        self.gridLayout_18 = QtWidgets.QGridLayout(self.tab_ExampleImgs_pop)
        self.gridLayout_18.setObjectName("gridLayout_18")
        self.splitter_5_pop = QtWidgets.QSplitter(self.tab_ExampleImgs_pop)
        self.splitter_5_pop.setOrientation(QtCore.Qt.Vertical)
        self.splitter_5_pop.setObjectName("splitter_5_pop")
        self.layoutWidget_6 = QtWidgets.QWidget(self.splitter_5_pop)
        self.layoutWidget_6.setObjectName("layoutWidget_6")
        self.horizontalLayout_ExampleImgs_pop = QtWidgets.QHBoxLayout(self.layoutWidget_6)
        self.horizontalLayout_ExampleImgs_pop.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_ExampleImgs_pop.setObjectName("horizontalLayout_ExampleImgs_pop")
        self.comboBox_ShowTrainOrValid_pop = QtWidgets.QComboBox(self.layoutWidget_6)
        self.comboBox_ShowTrainOrValid_pop.setObjectName("comboBox_ShowTrainOrValid_pop")
        self.horizontalLayout_ExampleImgs_pop.addWidget(self.comboBox_ShowTrainOrValid_pop)
        self.comboBox_ShowWOrWoAug_pop = QtWidgets.QComboBox(self.layoutWidget_6)
        self.comboBox_ShowWOrWoAug_pop.setObjectName("comboBox_ShowWOrWoAug_pop")
        self.horizontalLayout_ExampleImgs_pop.addWidget(self.comboBox_ShowWOrWoAug_pop)
        self.label_ShowIndex_pop = QtWidgets.QLabel(self.layoutWidget_6)
        self.label_ShowIndex_pop.setObjectName("label_ShowIndex_pop")
        self.label_ShowIndex_pop.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.horizontalLayout_ExampleImgs_pop.addWidget(self.label_ShowIndex_pop)
        self.spinBox_ShowIndex_pop = QtWidgets.QSpinBox(self.layoutWidget_6)
        self.spinBox_ShowIndex_pop.setObjectName("spinBox_ShowIndex_pop")
        self.horizontalLayout_ExampleImgs_pop.addWidget(self.spinBox_ShowIndex_pop)
        self.pushButton_ShowExamleImgs_pop = QtWidgets.QPushButton(self.layoutWidget_6)
        self.pushButton_ShowExamleImgs_pop.setObjectName("pushButton_ShowExamleImgs_pop")
        self.horizontalLayout_ExampleImgs_pop.addWidget(self.pushButton_ShowExamleImgs_pop)
        self.widget_ViewImages_pop = QtWidgets.QWidget(self.splitter_5_pop)
        self.widget_ViewImages_pop.setObjectName("widget_ViewImages_pop")
        self.gridLayout_18.addWidget(self.splitter_5_pop, 0, 0, 1, 1)
        self.tabWidget_DefineModel_pop.addTab(self.tab_ExampleImgs_pop, "")
        self.tab_expertMode_pop = QtWidgets.QWidget()
        #self.tab_expertMode_pop.setEnabled(True)
        self.tab_expertMode_pop.setObjectName("tab_expertMode_pop")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.tab_expertMode_pop)
        self.gridLayout_2.setObjectName("gridLayout_2")
        
        self.groupBox_expertMode_pop = QtWidgets.QGroupBox(self.tab_expertMode_pop)
        self.groupBox_expertMode_pop.setEnabled(True)
        self.groupBox_expertMode_pop.setCheckable(True)
        self.groupBox_expertMode_pop.setChecked(True)
        self.groupBox_expertMode_pop.setObjectName("groupBox_expertMode_pop")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox_expertMode_pop)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.scrollArea_expertMode_pop = QtWidgets.QScrollArea(self.groupBox_expertMode_pop)
        self.scrollArea_expertMode_pop.setEnabled(True)
        self.scrollArea_expertMode_pop.setWidgetResizable(True)
        self.scrollArea_expertMode_pop.setObjectName("scrollArea_expertMode_pop")
        self.scrollAreaWidgetContents_pop = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_pop.setGeometry(QtCore.QRect(0, -186, 697, 505))
        self.scrollAreaWidgetContents_pop.setObjectName("scrollAreaWidgetContents_pop")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents_pop)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.groupBox_modelKerasFit_pop = QtWidgets.QGroupBox(self.scrollAreaWidgetContents_pop)
        self.groupBox_modelKerasFit_pop.setObjectName("groupBox_modelKerasFit_pop")
        self.gridLayout_11 = QtWidgets.QGridLayout(self.groupBox_modelKerasFit_pop)
        self.gridLayout_11.setObjectName("gridLayout_11")
        self.label_batchSize_pop = QtWidgets.QLabel(self.groupBox_modelKerasFit_pop)
        self.label_batchSize_pop.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_batchSize_pop.setObjectName("label_batchSize_pop")
        self.gridLayout_11.addWidget(self.label_batchSize_pop, 0, 0, 1, 1)
        self.spinBox_batchSize = QtWidgets.QSpinBox(self.groupBox_modelKerasFit_pop)
        self.spinBox_batchSize.setMinimum(1)
        self.spinBox_batchSize.setMaximum(999999999)
        self.spinBox_batchSize.setProperty("value", 32)
        self.spinBox_batchSize.setObjectName("spinBox_batchSize")
        self.gridLayout_11.addWidget(self.spinBox_batchSize, 0, 1, 1, 1)
        self.label_epochs_pop = QtWidgets.QLabel(self.groupBox_modelKerasFit_pop)
        self.label_epochs_pop.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_epochs_pop.setObjectName("label_epochs_pop")
        self.gridLayout_11.addWidget(self.label_epochs_pop, 0, 2, 1, 1)
        self.spinBox_epochs = QtWidgets.QSpinBox(self.groupBox_modelKerasFit_pop)
        self.spinBox_epochs.setMinimum(1)
        self.spinBox_epochs.setMaximum(999999999)
        self.spinBox_epochs.setObjectName("spinBox_epochs")
        self.gridLayout_11.addWidget(self.spinBox_epochs, 0, 3, 1, 1)
        self.gridLayout_4.addWidget(self.groupBox_modelKerasFit_pop, 0, 0, 1, 1)
        self.groupBox_regularization_pop = QtWidgets.QGroupBox(self.scrollAreaWidgetContents_pop)
        self.groupBox_regularization_pop.setObjectName("groupBox_regularization_pop")
        self.gridLayout_12 = QtWidgets.QGridLayout(self.groupBox_regularization_pop)
        self.gridLayout_12.setObjectName("gridLayout_12")
        self.horizontalLayout_43_pop = QtWidgets.QHBoxLayout()
        self.horizontalLayout_43_pop.setObjectName("horizontalLayout_43_pop")
        self.checkBox_trainLastNOnly_pop = QtWidgets.QCheckBox(self.groupBox_regularization_pop)
        self.checkBox_trainLastNOnly_pop.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.checkBox_trainLastNOnly_pop.setCheckable(True)
        self.checkBox_trainLastNOnly_pop.setObjectName("checkBox_trainLastNOnly_pop")
        self.horizontalLayout_43_pop.addWidget(self.checkBox_trainLastNOnly_pop)
        self.spinBox_trainLastNOnly_pop = QtWidgets.QSpinBox(self.groupBox_regularization_pop)
        self.spinBox_trainLastNOnly_pop.setEnabled(False)
        self.spinBox_trainLastNOnly_pop.setMaximum(9999)
        self.spinBox_trainLastNOnly_pop.setObjectName("spinBox_trainLastNOnly_pop")
        self.horizontalLayout_43_pop.addWidget(self.spinBox_trainLastNOnly_pop)
        self.checkBox_trainDenseOnly_pop = QtWidgets.QCheckBox(self.groupBox_regularization_pop)
        self.checkBox_trainDenseOnly_pop.setObjectName("checkBox_trainDenseOnly_pop")
        self.horizontalLayout_43_pop.addWidget(self.checkBox_trainDenseOnly_pop)
        self.gridLayout_12.addLayout(self.horizontalLayout_43_pop, 0, 0, 1, 1)
        self.horizontalLayout_3_pop = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3_pop.setObjectName("horizontalLayout_3_pop")
        self.checkBox_dropout_pop = QtWidgets.QCheckBox(self.groupBox_regularization_pop)
        self.checkBox_dropout_pop.setObjectName("checkBox_dropout_pop")
        self.horizontalLayout_3_pop.addWidget(self.checkBox_dropout_pop)
        self.lineEdit_dropout_pop = QtWidgets.QLineEdit(self.groupBox_regularization_pop)
        self.lineEdit_dropout_pop.setEnabled(False)
        self.lineEdit_dropout_pop.setObjectName("lineEdit_dropout_pop")
        self.horizontalLayout_3_pop.addWidget(self.lineEdit_dropout_pop)
        self.gridLayout_12.addLayout(self.horizontalLayout_3_pop, 1, 0, 1, 1)
        self.horizontalLayout_pTr_pop = QtWidgets.QHBoxLayout()
        self.horizontalLayout_pTr_pop.setObjectName("horizontalLayout_pTr_pop")
#        self.checkBox_pTr_pop = QtWidgets.QCheckBox(self.groupBox_regularization_pop)
#        self.checkBox_pTr_pop.setObjectName("checkBox_pTr_pop")
#        self.horizontalLayout_pTr_pop.addWidget(self.checkBox_pTr_pop)
#        self.lineEdit_pTr_pop = QtWidgets.QLineEdit(self.groupBox_regularization_pop)
#        self.lineEdit_pTr_pop.setEnabled(False)
#        self.lineEdit_pTr_pop.setObjectName("lineEdit_pTr_pop")
#        self.horizontalLayout_pTr_pop.addWidget(self.lineEdit_pTr_pop)
#        self.pushButton_pTr_pop = QtWidgets.QPushButton(self.groupBox_regularization_pop)
#        self.pushButton_pTr_pop.setEnabled(False)
#        self.pushButton_pTr_pop.setMaximumSize(QtCore.QSize(40, 16777215))
#        self.pushButton_pTr_pop.setObjectName("pushButton_pTr_pop")
#        self.horizontalLayout_pTr_pop.addWidget(self.pushButton_pTr_pop)
        self.gridLayout_12.addLayout(self.horizontalLayout_pTr_pop, 2, 0, 1, 1)
        self.gridLayout_4.addWidget(self.groupBox_regularization_pop, 3, 0, 1, 1)       
        
        self.groupBox_lossOptimizer = QtWidgets.QGroupBox(self.scrollAreaWidgetContents_pop)
        self.groupBox_lossOptimizer.setObjectName("groupBox_lossOptimizer")
        self.gridLayout_14 = QtWidgets.QGridLayout(self.groupBox_lossOptimizer)
        self.gridLayout_14.setObjectName("gridLayout_14")
        self.pushButton_optimizer_pop = QtWidgets.QPushButton(self.groupBox_lossOptimizer)
        self.pushButton_optimizer_pop.setEnabled(False)
        self.pushButton_optimizer_pop.setMaximumSize(QtCore.QSize(40, 16777215))
        self.pushButton_optimizer_pop.setObjectName("pushButton_optimizer_pop")
        self.gridLayout_14.addWidget(self.pushButton_optimizer_pop, 0, 4, 1, 1)
        self.checkBox_lossW = QtWidgets.QCheckBox(self.groupBox_lossOptimizer)
        self.checkBox_lossW.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.checkBox_lossW.setObjectName("checkBox_lossW")
        self.gridLayout_14.addWidget(self.checkBox_lossW, 1, 0, 1, 1)
        self.checkBox_expt_loss_pop = QtWidgets.QCheckBox(self.groupBox_lossOptimizer)
        self.checkBox_expt_loss_pop.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.checkBox_expt_loss_pop.setObjectName("checkBox_expt_loss_pop")
        self.gridLayout_14.addWidget(self.checkBox_expt_loss_pop, 0, 0, 1, 1)
        self.comboBox_expt_loss_pop = QtWidgets.QComboBox(self.groupBox_lossOptimizer)
        self.comboBox_expt_loss_pop.setEnabled(False)
        self.comboBox_expt_loss_pop.setObjectName("comboBox_expt_loss_pop")
        self.comboBox_expt_loss_pop.addItem("")
        self.comboBox_expt_loss_pop.addItem("")
        self.comboBox_expt_loss_pop.addItem("")
        self.comboBox_expt_loss_pop.addItem("")
        self.comboBox_expt_loss_pop.addItem("")
        self.comboBox_expt_loss_pop.addItem("")
        self.comboBox_expt_loss_pop.addItem("")
        self.comboBox_expt_loss_pop.addItem("")
        self.comboBox_expt_loss_pop.addItem("")
        self.comboBox_expt_loss_pop.addItem("")
        self.comboBox_expt_loss_pop.addItem("")
        self.comboBox_expt_loss_pop.addItem("")
        self.comboBox_expt_loss_pop.addItem("")
        self.comboBox_expt_loss_pop.addItem("")
        self.comboBox_expt_loss_pop.addItem("")
        self.comboBox_expt_loss_pop.addItem("")
        self.gridLayout_14.addWidget(self.comboBox_expt_loss_pop, 0, 1, 1, 1)
        self.checkBox_optimizer_pop = QtWidgets.QCheckBox(self.groupBox_lossOptimizer)
        self.checkBox_optimizer_pop.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.checkBox_optimizer_pop.setObjectName("checkBox_optimizer_pop")
        self.gridLayout_14.addWidget(self.checkBox_optimizer_pop, 0, 2, 1, 1)
        self.comboBox_optimizer = QtWidgets.QComboBox(self.groupBox_lossOptimizer)
        self.comboBox_optimizer.setEnabled(False)
        self.comboBox_optimizer.setObjectName("comboBox_optimizer")
        self.comboBox_optimizer.addItem("")
        self.comboBox_optimizer.addItem("")
        self.comboBox_optimizer.addItem("")
        self.comboBox_optimizer.addItem("")
        self.comboBox_optimizer.addItem("")
        self.comboBox_optimizer.addItem("")
        self.comboBox_optimizer.addItem("")
        self.gridLayout_14.addWidget(self.comboBox_optimizer, 0, 3, 1, 1)
        self.lineEdit_lossW = QtWidgets.QLineEdit(self.groupBox_lossOptimizer)
        self.lineEdit_lossW.setEnabled(False)
        self.lineEdit_lossW.setObjectName("lineEdit_lossW")
        self.gridLayout_14.addWidget(self.lineEdit_lossW, 1, 1, 1, 3)
        self.pushButton_lossW = QtWidgets.QPushButton(self.groupBox_lossOptimizer)
        self.pushButton_lossW.setEnabled(False)
        self.pushButton_lossW.setMaximumSize(QtCore.QSize(40, 16777215))
        self.pushButton_lossW.setObjectName("pushButton_lossW")
        self.gridLayout_14.addWidget(self.pushButton_lossW, 1, 4, 1, 1)
        self.gridLayout_4.addWidget(self.groupBox_lossOptimizer, 1, 0, 1, 1)
        
        self.groupBox_learningRate_pop = QtWidgets.QGroupBox(self.scrollAreaWidgetContents_pop)
        self.groupBox_learningRate_pop.setEnabled(True)
        self.groupBox_learningRate_pop.setCheckable(True)
        self.groupBox_learningRate_pop.setChecked(False)
        self.groupBox_learningRate_pop.setObjectName("groupBox_learningRate_pop")
        self.gridLayout_16 = QtWidgets.QGridLayout(self.groupBox_learningRate_pop)
        self.gridLayout_16.setObjectName("gridLayout_16")
        self.radioButton_LrConst = QtWidgets.QRadioButton(self.groupBox_learningRate_pop)
        self.radioButton_LrConst.setChecked(True)
        self.radioButton_LrConst.setObjectName("radioButton_LrConst")
        self.gridLayout_16.addWidget(self.radioButton_LrConst, 0, 0, 1, 1)
        
        self.label_LrConst_pop = QtWidgets.QLabel(self.groupBox_learningRate_pop)
        self.label_LrConst_pop.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_LrConst_pop.setObjectName("label_LrConst_pop")
        self.gridLayout_16.addWidget(self.label_LrConst_pop, 0, 1, 1, 1)
        self.doubleSpinBox_learningRate = QtWidgets.QDoubleSpinBox(self.groupBox_learningRate_pop)
        self.doubleSpinBox_learningRate.setEnabled(False)
        self.doubleSpinBox_learningRate.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.doubleSpinBox_learningRate.setDecimals(6)
        self.doubleSpinBox_learningRate.setMaximum(999.0)
        self.doubleSpinBox_learningRate.setSingleStep(0.0001)
        self.doubleSpinBox_learningRate.setProperty("value", 0.001)
        self.doubleSpinBox_learningRate.setObjectName("doubleSpinBox_learningRate")
        self.gridLayout_16.addWidget(self.doubleSpinBox_learningRate, 0, 2, 1, 1)
        self.radioButton_LrCycl = QtWidgets.QRadioButton(self.groupBox_learningRate_pop)
        self.radioButton_LrCycl.setObjectName("radioButton_LrCycl")
        self.gridLayout_16.addWidget(self.radioButton_LrCycl, 1, 0, 1, 1)
        self.label_cycLrMin = QtWidgets.QLabel(self.groupBox_learningRate_pop)
        self.label_cycLrMin.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_cycLrMin.setObjectName("label_cycLrMin")
        self.gridLayout_16.addWidget(self.label_cycLrMin, 1, 1, 1, 1)
        self.lineEdit_cycLrMin = QtWidgets.QLineEdit(self.groupBox_learningRate_pop)
        self.lineEdit_cycLrMin.setObjectName("lineEdit_cycLrMin")
        self.lineEdit_cycLrMin.setEnabled(False)
        validator = QtGui.QRegExpValidator(QtCore.QRegExp("^[0-9 . , e -]+$")) #validator allows numbers, dots, commas, e and -
        self.lineEdit_cycLrMin.setValidator(validator)
        self.gridLayout_16.addWidget(self.lineEdit_cycLrMin, 1, 2, 1, 1)
        self.lineEdit_cycLrMax = QtWidgets.QLineEdit(self.groupBox_learningRate_pop)
        self.lineEdit_cycLrMax.setObjectName("lineEdit_cycLrMax")
        self.lineEdit_cycLrMax.setEnabled(False)
        self.lineEdit_cycLrMax.setValidator(validator)
        self.gridLayout_16.addWidget(self.lineEdit_cycLrMax, 1, 3, 1, 1)
        self.label_cycLrMethod = QtWidgets.QLabel(self.groupBox_learningRate_pop)
        self.label_cycLrMethod.setObjectName("label_cycLrMethod")
        self.label_cycLrMethod.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.gridLayout_16.addWidget(self.label_cycLrMethod, 1, 4, 1, 1)
        self.comboBox_cycLrMethod = QtWidgets.QComboBox(self.groupBox_learningRate_pop)
        self.comboBox_cycLrMethod.setEnabled(False)
        self.comboBox_cycLrMethod.setMinimumSize(QtCore.QSize(80, 0))
        self.comboBox_cycLrMethod.setObjectName("comboBox_cycLrMethod")
        self.comboBox_cycLrMethod.addItem("")
        self.comboBox_cycLrMethod.addItem("")
        self.comboBox_cycLrMethod.addItem("")
        self.gridLayout_16.addWidget(self.comboBox_cycLrMethod, 1, 6, 1, 1)
        self.pushButton_cycLrPopup = QtWidgets.QPushButton(self.groupBox_learningRate_pop)
        self.pushButton_cycLrPopup.setEnabled(False)
        self.pushButton_cycLrPopup.setMaximumSize(QtCore.QSize(50, 16777215))
        self.pushButton_cycLrPopup.setObjectName("pushButton_cycLrPopup")
        self.gridLayout_16.addWidget(self.pushButton_cycLrPopup, 1, 7, 1, 1)
        self.radioButton_LrExpo = QtWidgets.QRadioButton(self.groupBox_learningRate_pop)
        self.radioButton_LrExpo.setObjectName("radioButton_LrExpo")
        self.gridLayout_16.addWidget(self.radioButton_LrExpo, 2, 0, 2, 1)
        self.label_expDecInitLr = QtWidgets.QLabel(self.groupBox_learningRate_pop)
        self.label_expDecInitLr.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_expDecInitLr.setObjectName("label_expDecInitLr")
        self.gridLayout_16.addWidget(self.label_expDecInitLr, 2, 1, 1, 1)
        self.doubleSpinBox_expDecInitLr = QtWidgets.QDoubleSpinBox(self.groupBox_learningRate_pop)
        self.doubleSpinBox_expDecInitLr.setEnabled(False)
        self.doubleSpinBox_expDecInitLr.setMaximumSize(QtCore.QSize(63, 16777215))
        self.doubleSpinBox_expDecInitLr.setDecimals(6)
        self.doubleSpinBox_expDecInitLr.setSingleStep(0.0001)
        self.doubleSpinBox_expDecInitLr.setProperty("value", 0.001)
        self.doubleSpinBox_expDecInitLr.setObjectName("doubleSpinBox_expDecInitLr")
        self.gridLayout_16.addWidget(self.doubleSpinBox_expDecInitLr, 2, 2, 2, 1)
        self.label_expDecSteps = QtWidgets.QLabel(self.groupBox_learningRate_pop)
        self.label_expDecSteps.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_expDecSteps.setObjectName("label_expDecSteps")
        self.gridLayout_16.addWidget(self.label_expDecSteps, 2, 3, 1, 1)
        self.spinBox_expDecSteps = QtWidgets.QSpinBox(self.groupBox_learningRate_pop)
        self.spinBox_expDecSteps.setEnabled(False)
        self.spinBox_expDecSteps.setMaximumSize(QtCore.QSize(63, 16777215))
        self.spinBox_expDecSteps.setMaximum(999999999)
        self.spinBox_expDecSteps.setObjectName("spinBox_expDecSteps")
        self.gridLayout_16.addWidget(self.spinBox_expDecSteps, 2, 4, 2, 2)
        self.label_expDecRate = QtWidgets.QLabel(self.groupBox_learningRate_pop)
        self.label_expDecRate.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_expDecRate.setObjectName("label_expDecRate")
        self.gridLayout_16.addWidget(self.label_expDecRate, 2, 6, 1, 1)
        self.doubleSpinBox_expDecRate = QtWidgets.QDoubleSpinBox(self.groupBox_learningRate_pop)
        self.doubleSpinBox_expDecRate.setEnabled(False)
        self.doubleSpinBox_expDecRate.setMaximumSize(QtCore.QSize(63, 16777215))
        self.doubleSpinBox_expDecRate.setDecimals(6)
        self.doubleSpinBox_expDecRate.setMaximum(1.0)
        self.doubleSpinBox_expDecRate.setSingleStep(0.01)
        self.doubleSpinBox_expDecRate.setProperty("value", 0.96)
        self.doubleSpinBox_expDecRate.setObjectName("doubleSpinBox_expDecRate")
        self.gridLayout_16.addWidget(self.doubleSpinBox_expDecRate, 2, 7, 1, 1)
        self.line = QtWidgets.QFrame(self.groupBox_learningRate_pop)
        self.line.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.gridLayout_16.addWidget(self.line, 4, 5, 1, 1)
        self.pushButton_LR_finder = QtWidgets.QPushButton(self.groupBox_learningRate_pop)
        self.pushButton_LR_finder.setObjectName("pushButton_LR_finder")
        self.gridLayout_16.addWidget(self.pushButton_LR_finder, 4, 6, 1, 1)
        self.pushButton_LR_plot = QtWidgets.QPushButton(self.groupBox_learningRate_pop)
        self.pushButton_LR_plot.setObjectName("pushButton_LR_plot")
        self.gridLayout_16.addWidget(self.pushButton_LR_plot, 4, 7, 1, 1)
        self.gridLayout_4.addWidget(self.groupBox_learningRate_pop, 2, 0, 1, 1)
        self.scrollArea_expertMode_pop.setWidget(self.scrollAreaWidgetContents_pop)
        self.gridLayout_3.addWidget(self.scrollArea_expertMode_pop, 0, 0, 1, 1)
        self.gridLayout_2.addWidget(self.groupBox_expertMode_pop, 0, 0, 1, 1)
        
        
        
        
        
        
        
        
        
        
        
        
        
        self.tabWidget_DefineModel_pop.addTab(self.tab_expertMode_pop, "")
        self.gridLayout_3_pop.addWidget(self.tabWidget_DefineModel_pop, 1, 0, 1, 1)
        self.verticalLayout_3_pop.addWidget(self.splitter_pop)
        self.verticalLayout_4_pop.addLayout(self.verticalLayout_3_pop)
        self.gridLayout_slider_pop.addLayout(self.verticalLayout_4_pop, 0, 0, 1, 1)



        ######################ICONS############################################        
        os.path.join(dir_root,"art",Default_dict["Icon theme"],"color_mode.png")

        self.label_colorModeIcon_pop.setPixmap(QtGui.QPixmap(os.path.join(dir_root,"art",Default_dict["Icon theme"],"color_mode.png")))
        self.label_NormalizationIcon_pop.setPixmap(QtGui.QPixmap(os.path.join(dir_root,"art",Default_dict["Icon theme"],"normalization.png")))
        self.label_Crop_NrEpochsIcon_pop.setPixmap(QtGui.QPixmap(os.path.join(dir_root,"art",Default_dict["Icon theme"],"nr_epochs.png")))
        self.label_zoomIcon.setPixmap(QtGui.QPixmap(os.path.join(dir_root,"art",Default_dict["Icon theme"],"zoom_order.png")))

        self.label_ModelGeomIcon_pop.setPixmap(QtGui.QPixmap(os.path.join(dir_root,"art",Default_dict["Icon theme"],"model_architecture.png")))
        self.label_padIcon_pop.setPixmap(QtGui.QPixmap(os.path.join(dir_root,"art",Default_dict["Icon theme"],"padding.png")))
        
        self.label_CropIcon_pop.setPixmap(QtGui.QPixmap(os.path.join(dir_root,"art",Default_dict["Icon theme"],"cropping.png")))
        self.label_Crop_pop.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(os.path.join(dir_root,"art",Default_dict["Icon theme"],"gpu.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.radioButton_gpu_pop.setIcon(icon)


        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(os.path.join(dir_root,"art",Default_dict["Icon theme"],"cpu.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.radioButton_cpu_pop.setIcon(icon1)
        

        self.checkBox_ApplyNextEpoch.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"thumb.png")))

        self.checkBox_saveEpoch_pop.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"save_epoch.png")))
        self.pushButton_Pause_pop.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"pause.png")))
        self.pushButton_Stop_pop.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"stop.png")))

        self.checkBox_HorizFlip_pop.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"horizontal_flip.png")))
        self.checkBox_VertFlip_pop.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"vertical_flip.png")))
        self.label_Rotation_pop.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"rotation.png")))
        self.label_Rotation_pop.setChecked(True)
        self.label_Rotation_pop.stateChanged.connect(self.keras_changed_rotation_pop)
        self.label_width_shift_pop.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"width_shift.png")))
        self.label_width_shift_pop.setChecked(True)
        self.label_width_shift_pop.stateChanged.connect(self.keras_changed_width_shift_pop)
        self.label_height_shift_pop.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"height_shift.png")))
        self.label_height_shift_pop.setChecked(True)
        self.label_height_shift_pop.stateChanged.connect(self.keras_changed_height_shift_pop)
        self.label_zoom_pop.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"zoom.png")))
        self.label_zoom_pop.setChecked(True)
        self.label_zoom_pop.stateChanged.connect(self.keras_changed_zoom_pop)
        self.label_shear_pop.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"shear.png")))
        self.label_shear_pop.setChecked(True)
        self.label_shear_pop.stateChanged.connect(self.keras_changed_shear_pop)
        self.label_Plus_pop.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"brightness_plus.png")))
        self.label_Plus_pop.setChecked(True)
        self.label_Plus_pop.stateChanged.connect(self.keras_changed_brightplus_pop)
        self.label_Mult_pop.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"brightness_mult.png")))
        self.label_Mult_pop.setChecked(True)
        self.label_Mult_pop.stateChanged.connect(self.keras_changed_brightmult_pop)
        self.label_GaussianNoiseMean_pop.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"gaussian_noise_mean.png")))
        self.label_GaussianNoiseMean_pop.setChecked(True)
        self.label_GaussianNoiseMean_pop.stateChanged.connect(self.keras_changed_noiseMean_pop)
        self.label_GaussianNoiseScale_pop.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"gaussian_noise_scale.png")))
        self.label_GaussianNoiseScale_pop.setChecked(True)
        self.label_GaussianNoiseScale_pop.stateChanged.connect(self.keras_changed_noiseScale_pop)
        self.checkBox_contrast_pop.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"contrast.png")))
        self.checkBox_contrast_pop.stateChanged.connect(self.keras_changed_contrast_pop)
        self.checkBox_saturation_pop.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"saturation.png")))
        self.checkBox_saturation_pop.stateChanged.connect(self.keras_changed_saturation_pop)
        self.checkBox_hue_pop.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"hue.png")))
        self.checkBox_hue_pop.stateChanged.connect(self.keras_changed_hue_pop)

        self.checkBox_avgBlur_pop.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"average_blur.png")))
        #self.checkBox_avgBlur_pop.stateChanged.connect(self.changed_averageBlur_pop)
        self.checkBox_gaussBlur_pop.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"gaussian_blur.png")))
        #self.checkBox_gaussBlur_pop.stateChanged.connect(self.changed_gaussBlur_pop)
        self.checkBox_motionBlur_pop.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"motion_blur.png")))
        #self.checkBox_motionBlur_pop.stateChanged.connect(self.changed_motionBlur_pop)



        validator = QtGui.QRegExpValidator(QtCore.QRegExp("^[0-9 . ,]+$")) #validator allows numbers, dots and commas
        #aternatively, I could use "^[0-9 . , \[ \] ]+$" - this would also allow the user to put the brackets. But why? I just do it in the program
        self.lineEdit_dropout_pop.setValidator(validator)        

        ############Icons Expert tab########
        #self.pushButton_LR_finder.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"lr_screen.png")))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(os.path.join(dir_root,"art",Default_dict["Icon theme"],"lr_screen.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_LR_finder.setIcon(icon)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(os.path.join(dir_root,"art",Default_dict["Icon theme"],"lr_plot.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_LR_plot.setIcon(icon)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(os.path.join(dir_root,"art",Default_dict["Icon theme"],"lr_const.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.radioButton_LrConst.setIcon(icon)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(os.path.join(dir_root,"art",Default_dict["Icon theme"],"lr_cycle.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.radioButton_LrCycl.setIcon(icon)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(os.path.join(dir_root,"art",Default_dict["Icon theme"],"lr_exponential.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.radioButton_LrExpo.setIcon(icon)



        #####################Some manual settings##############################
        #######################################################################        
        ###########################Variables###################################
        self.Histories = [] #List container for the fitting histories, that are produced by the keras.fit function that is controlled by this popup
        self.RealTime_Acc,self.RealTime_ValAcc,self.RealTime_Loss,self.RealTime_ValLoss = [],[],[],[]
        self.RealTime_OtherMetrics = {} #provide dictionary where AID can save all other metrics in case there are some (like precision...)
        self.X_batch_aug = []#list for storing augmented image, created by some parallel processes
        self.threadpool_quad = QtCore.QThreadPool()#Threadpool for image augmentation
        self.threadpool_quad.setMaxThreadCount(4)#Maximum 4 threads
        self.threadpool_quad_count = 0 #count nr. of threads in queue; 
        self.clr_settings = {} #variable to store step_size and gamma, will be filled with information when starting to fit
        self.optimizer_settings = {} #dict to store advanced optimizer settings
        
        self.epoch_counter = 0 #Counts the nr. of epochs
        self.tableWidget_HistoryInfo_pop.setMinimumSize(QtCore.QSize(0, 100))
        self.tableWidget_HistoryInfo_pop.setMaximumSize(QtCore.QSize(16777215, 140))
        self.tableWidget_HistoryInfo_pop.setColumnCount(7)
        self.tableWidget_HistoryInfo_pop.setRowCount(0)
        self.spinBox_imagecrop_pop.setMinimum(1)
        self.spinBox_imagecrop_pop.setMaximum(9E8)

        #self.comboBox_colorMode_pop.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.label_Normalization_pop.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.label_Crop_NrEpochs_pop.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        self.spinBox_RefreshAfterEpochs_pop.setMinimum(1)
        self.spinBox_RefreshAfterEpochs_pop.setMaximum(9E8)
        self.spinBox_RefreshAfterNrEpochs_pop.setMinimum(1)
        self.spinBox_RefreshAfterNrEpochs_pop.setMaximum(9E8)
        self.spinBox_PlusLower_pop.setMinimum(-255)
        self.spinBox_PlusLower_pop.setMaximum(255)
        self.spinBox_PlusLower_pop.setSingleStep(1)
        self.spinBox_PlusUpper_pop.setMinimum(-255)
        self.spinBox_PlusUpper_pop.setMaximum(255)
        self.spinBox_PlusUpper_pop.setSingleStep(1)
        self.doubleSpinBox_MultLower_pop.setMinimum(0)
        self.doubleSpinBox_MultLower_pop.setMaximum(10)
        self.doubleSpinBox_MultLower_pop.setSingleStep(0.1)
        self.doubleSpinBox_MultUpper_pop.setMinimum(0)
        self.doubleSpinBox_MultUpper_pop.setMaximum(10)
        self.doubleSpinBox_MultUpper_pop.setSingleStep(0.1)
        self.doubleSpinBox_GaussianNoiseMean_pop.setMinimum(-255)
        self.doubleSpinBox_GaussianNoiseMean_pop.setMaximum(255) 
        self.doubleSpinBox_GaussianNoiseMean_pop.setSingleStep(0.1)
        self.doubleSpinBox_GaussianNoiseScale_pop.setMinimum(0)
        self.doubleSpinBox_GaussianNoiseScale_pop.setMaximum(99.9)
        self.doubleSpinBox_GaussianNoiseScale_pop.setSingleStep(0.1)

        self.spinBox_avgBlurMin_pop.setMinimum(0)
        self.spinBox_avgBlurMin_pop.setMaximum(255)
        self.spinBox_avgBlurMax_pop.setMinimum(0)
        self.spinBox_avgBlurMax_pop.setMaximum(255)
        self.spinBox_gaussBlurMin_pop.setMinimum(0)
        self.spinBox_gaussBlurMin_pop.setMaximum(255)
        self.spinBox_gaussBlurMax_pop.setMinimum(0)
        self.spinBox_gaussBlurMax_pop.setMaximum(255)


        self.comboBox_ShowTrainOrValid_pop.addItems(["Training","Validation"])       
        self.comboBox_ShowWOrWoAug_pop.addItems(["With Augmentation","Original image"])    


#        self.groupBox_expertMode_pop.setEnabled(True)
#        self.groupBox_expertMode_pop.setCheckable(True)
#        self.groupBox_expertMode_pop.setChecked(False)
#        self.scrollArea_expertMode_pop.setWidgetResizable(True)
  
        #Adjust some QObjects manually
        self.spinBox_batchSize.setMinimum(1)       
        self.spinBox_batchSize.setMaximum(1E6)       
        self.spinBox_batchSize.setValue(32)       
        self.spinBox_epochs.setMinimum(1)       
        self.spinBox_epochs.setMaximum(1E6)       
        self.spinBox_epochs.setValue(1)       
        self.doubleSpinBox_learningRate.setDecimals(9)
        self.doubleSpinBox_learningRate.setMinimum(0.0)       
        self.doubleSpinBox_learningRate.setMaximum(1E6)       
        self.doubleSpinBox_learningRate.setValue(0.001)       
        self.doubleSpinBox_learningRate.setSingleStep(0.0001)
        self.spinBox_trainLastNOnly_pop.setMinimum(0)       
        self.spinBox_trainLastNOnly_pop.setMaximum(1E6)       
        self.spinBox_trainLastNOnly_pop.setValue(0)    
        self.checkBox_trainDenseOnly_pop.setChecked(False)

        self.spinBox_NrEpochs.setMinimum(1)
        self.spinBox_NrEpochs.setMaximum(9E8)

        self.spinBox_realTimeEpochs.setSingleStep(1)
        self.spinBox_realTimeEpochs.setMinimum(1)
        self.spinBox_realTimeEpochs.setMaximum(9999999)
        self.spinBox_realTimeEpochs.setValue(250)        
        self.pushButton_Pause_pop.setMinimumSize(QtCore.QSize(60, 30))
        self.pushButton_Pause_pop.setMaximumSize(QtCore.QSize(60, 30))
        self.pushButton_Stop_pop.setMinimumSize(QtCore.QSize(60, 30))
        self.pushButton_Stop_pop.setMaximumSize(QtCore.QSize(60, 30))

        #######################################################################
        ######################Connections######################################
        self.doubleSpinBox_learningRate.setEnabled(False)
        self.spinBox_trainLastNOnly_pop.setEnabled(False)
        self.lineEdit_dropout_pop.setEnabled(False)
        self.pushButton_LR_finder.setEnabled(False)
        #self.pushButton_LR_plot.setEnabled(False)


        self.radioButton_LrConst.toggled['bool'].connect(self.doubleSpinBox_learningRate.setEnabled)
        self.radioButton_LrCycl.toggled['bool'].connect(self.lineEdit_cycLrMin.setEnabled)
        self.radioButton_LrCycl.toggled['bool'].connect(self.lineEdit_cycLrMax.setEnabled)
        self.radioButton_LrCycl.toggled['bool'].connect(self.comboBox_cycLrMethod.setEnabled)
        self.radioButton_LrCycl.toggled['bool'].connect(self.pushButton_cycLrPopup.setEnabled)
        self.radioButton_LrExpo.toggled['bool'].connect(self.doubleSpinBox_expDecInitLr.setEnabled)
        self.radioButton_LrExpo.toggled['bool'].connect(self.spinBox_expDecSteps.setEnabled)
        self.radioButton_LrExpo.toggled['bool'].connect(self.doubleSpinBox_expDecRate.setEnabled)
        
        self.groupBox_learningRate_pop.toggled['bool'].connect(self.doubleSpinBox_learningRate.setEnabled) 

        self.checkBox_expt_loss_pop.toggled['bool'].connect(self.comboBox_expt_loss_pop.setEnabled)
        self.checkBox_optimizer_pop.toggled['bool'].connect(self.comboBox_optimizer.setEnabled)
        self.checkBox_optimizer_pop.toggled['bool'].connect(self.pushButton_optimizer_pop.setEnabled)

        self.checkBox_trainLastNOnly_pop.toggled['bool'].connect(self.spinBox_trainLastNOnly_pop.setEnabled)
        self.checkBox_dropout_pop.toggled['bool'].connect(self.lineEdit_dropout_pop.setEnabled)

        self.checkBox_avgBlur_pop.clicked['bool'].connect(self.spinBox_avgBlurMin_pop.setEnabled)
        self.checkBox_avgBlur_pop.clicked['bool'].connect(self.spinBox_avgBlurMax_pop.setEnabled)
        self.checkBox_gaussBlur_pop.clicked['bool'].connect(self.spinBox_gaussBlurMin_pop.setEnabled)
        self.checkBox_gaussBlur_pop.clicked['bool'].connect(self.spinBox_gaussBlurMax_pop.setEnabled)
        self.checkBox_motionBlur_pop.clicked['bool'].connect(self.label_motionBlurKernel_pop.setEnabled)
        self.checkBox_motionBlur_pop.clicked['bool'].connect(self.lineEdit_motionBlurKernel_pop.setEnabled)
        self.checkBox_motionBlur_pop.clicked['bool'].connect(self.label_motionBlurAngle_pop.setEnabled)
        self.checkBox_motionBlur_pop.clicked['bool'].connect(self.lineEdit_motionBlurAngle_pop.setEnabled)
        self.checkBox_gaussBlur_pop.clicked['bool'].connect(self.label_gaussBlurMin_pop.setEnabled)
        self.checkBox_gaussBlur_pop.clicked['bool'].connect(self.label_gaussBlurMax_pop.setEnabled)
        self.checkBox_avgBlur_pop.clicked['bool'].connect(self.label_avgBlurMin_pop.setEnabled)
        self.checkBox_avgBlur_pop.clicked['bool'].connect(self.label_avgBlurMax_pop.setEnabled)

        self.checkBox_optimizer_pop.toggled['bool'].connect(self.comboBox_optimizer.setEnabled)
        self.checkBox_expt_loss_pop.toggled['bool'].connect(self.comboBox_expt_loss_pop.setEnabled)
        #self.comboBox_optimizer.currentTextChanged.connect(lambda: self.expert_optimizer_changed())
        self.checkBox_optimizer_pop.stateChanged.connect(self.expert_optimizer_off_pop)
        self.groupBox_learningRate_pop.toggled.connect(self.expert_learningrate_off_pop)
        self.checkBox_expt_loss_pop.stateChanged.connect(self.expert_loss_off_pop)
        self.groupBox_expertMode_pop.toggled.connect(self.expert_mode_off_pop)

        




        self.retranslateUi(Form)
        self.tabWidget_DefineModel_pop.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "Form", None))
        self.pushButton_UpdatePlot_pop.setText(_translate("Form", "Update Plot", None))
        self.checkBox_realTimePlotting_pop.setToolTip(_translate("Form", tooltips["checkBox_realTimePlotting_pop"], None))
        self.checkBox_realTimePlotting_pop.setText(_translate("Form", "Real-time plotting", None))
        self.label_realTimeEpochs_pop.setText(_translate("Form", "Nr. of epochs for RT", None))
        self.label_realTimeEpochs_pop.setToolTip(_translate("Form",tooltips["label_realTimeEpochs_pop"] , None))
        self.spinBox_realTimeEpochs.setToolTip(_translate("Form", tooltips["label_realTimeEpochs_pop"], None))


        self.groupBox_FittingInfo_pop.setTitle(_translate("Form", "Fitting Info", None))
        self.pushButton_saveTextWindow_pop.setText(_translate("Form", "Save text ", None))
        self.pushButton_clearTextWindow_pop.setToolTip(_translate("Form",tooltips["pushButton_clearTextWindow_pop"] , None))
        self.pushButton_clearTextWindow_pop.setText(_translate("Form", "Clear text", None))
        self.groupBox_ChangeModel_pop.setTitle(_translate("Form", "Change fitting parameters", None))
        self.label_ModelGeom_pop.setText(_translate("Form", "Model Architecture", None))
        self.label_ModelGeom_pop.setToolTip(_translate("Form", tooltips["comboBox_ModelSelection"], None))
        self.label_ModelGeomIcon_pop.setToolTip(_translate("Form", tooltips["comboBox_ModelSelection"], None))
        self.comboBox_ModelSelection_pop.setToolTip(_translate("Form", tooltips["comboBox_ModelSelection"], None))

        self.label_colorMode_pop.setToolTip(_translate("Form", "Color mode used for this model", None))
        self.label_colorMode_pop.setText(_translate("Form", "Color Mode", None))
        self.label_colorModeIcon_pop.setToolTip(_translate("Form", "Color mode used for this model", None))

        self.comboBox_colorMode_pop.setToolTip(_translate("Form", "Color mode used for this model", None))
        self.label_Normalization_pop.setToolTip(_translate("Form", tooltips["label_Normalization"], None))
        self.label_Normalization_pop.setText(_translate("Form", "Normalization", None))
        self.label_NormalizationIcon_pop.setToolTip(_translate("Form", tooltips["label_Normalization"], None))
        self.comboBox_Normalization_pop.setToolTip(_translate("Form", tooltips["label_Normalization"], None))
        
        self.label_zoomOrder.setText(_translate("Form", "Zoom order", None))
        
        self.label_Crop_pop.setToolTip(_translate("Form", tooltips["label_Crop"], None))
        self.label_Crop_pop.setText(_translate("Form", "Input image size", None))
        self.label_CropIcon_pop.setToolTip(_translate("Form", tooltips["label_Crop"], None))

        self.groupBox_system_pop.setTitle(_translate("Form", "Training", None))
        self.label_padIcon_pop.setToolTip(_translate("Form", tooltips["label_paddingMode"], None))
        self.comboBox_paddingMode_pop.setToolTip(_translate("Form", tooltips["label_paddingMode"], None))
        self.label_paddingMode_pop.setToolTip(_translate("Form", tooltips["label_paddingMode"], None))
        self.spinBox_imagecrop_pop.setToolTip(_translate("Form", tooltips["label_Crop"], None))
        self.label_Crop_NrEpochs_pop.setToolTip(_translate("Form", "Total number of training iterations", None))
        self.label_Crop_NrEpochs_pop.setText(_translate("Form", "Nr. epochs", None))
        self.spinBox_NrEpochs.setToolTip(_translate("Form", "Total number of training iterations", None))
        self.label_Crop_NrEpochsIcon_pop.setToolTip(_translate("Form", "Total number of training iterations", None))
        self.label_saveMetaEvery.setText(_translate("Form", "Save meta every (sec)", None))
        self.label_saveMetaEvery.setToolTip(_translate("Form", tooltips["label_saveMetaEvery"], None))
        self.spinBox_saveMetaEvery.setToolTip(_translate("Form", tooltips["label_saveMetaEvery"], None))

        self.radioButton_cpu_pop.setToolTip(_translate("MainWindow",tooltips["radioButton_cpu"] , None))
        self.comboBox_cpu_pop.setToolTip(_translate("MainWindow",tooltips["radioButton_cpu"] , None))
        self.radioButton_gpu_pop.setToolTip(_translate("MainWindow",tooltips["radioButton_gpu"] , None))
        self.comboBox_gpu_pop.setToolTip(_translate("MainWindow",tooltips["comboBox_gpu"] , None))

        self.label_memory_pop.setText(_translate("MainWindow", "Memory", None))
        self.label_memory_pop.setToolTip(_translate("MainWindow", tooltips["label_memory"],None))
        self.doubleSpinBox_memory_pop.setToolTip(_translate("MainWindow", tooltips["label_memory"],None))








        self.pushButton_modelname_pop.setToolTip(_translate("Form", tooltips["pushButton_modelname"], None))
        self.pushButton_modelname_pop.setText(_translate("Form", "Model path:", None))
        self.lineEdit_modelname_pop.setToolTip(_translate("Form", tooltips["pushButton_modelname"], None))
        self.pushButton_showModelSumm_pop.setText(_translate("Form", "Show model summary", None))
        self.pushButton_saveModelSumm_pop.setText(_translate("Form", "Save model summary", None))
        self.checkBox_ApplyNextEpoch.setToolTip(_translate("Form", tooltips["checkBox_ApplyNextEpoch"], None))
        self.checkBox_ApplyNextEpoch.setText(_translate("Form", "Apply at next epoch", None))
        self.checkBox_saveEpoch_pop.setToolTip(_translate("Form",tooltips["checkBox_saveEpoch_pop"] , None))
        self.checkBox_saveEpoch_pop.setText(_translate("Form", "Save Model", None))
        self.pushButton_Pause_pop.setToolTip(_translate("Form", tooltips["pushButton_Pause_pop"], None))
        self.pushButton_Pause_pop.setText(_translate("Form", " ", None))
        self.pushButton_Stop_pop.setToolTip(_translate("Form", tooltips["pushButton_Stop_pop"], None))
        self.pushButton_Stop_pop.setText(_translate("Form", "", None))
        self.tabWidget_DefineModel_pop.setTabText(self.tabWidget_DefineModel_pop.indexOf(self.tab_DefineModel_pop), _translate("Form", "Define Model", None))
        self.tab_KerasImgAug_pop.setToolTip(_translate("Form", tooltips["tab_kerasAug"], None))
        self.label_RefreshAfterEpochs_pop.setToolTip(_translate("Form", tooltips["spinBox_RefreshAfterEpochs"], None))
        self.label_RefreshAfterEpochs_pop.setText(_translate("Form", "Refresh after nr. epochs:", None))
        self.spinBox_RefreshAfterEpochs_pop.setToolTip(_translate("Form", tooltips["spinBox_RefreshAfterEpochs"], None))
        self.checkBox_HorizFlip_pop.setToolTip(_translate("Form", tooltips["checkBox_HorizFlip"], None))
        self.checkBox_HorizFlip_pop.setText(_translate("Form", "Horizontal flip", None))
        self.checkBox_VertFlip_pop.setToolTip(_translate("Form", tooltips["checkBox_VertFlip"], None))
        self.checkBox_VertFlip_pop.setText(_translate("Form", "Vertical flip", None))
        self.label_Rotation_pop.setToolTip(_translate("Form", tooltips["label_Rotation"], None))
        self.label_Rotation_pop.setText(_translate("Form", "Rotation", None))
        self.label_width_shift_pop.setToolTip(_translate("Form", tooltips["label_width_shift"], None))
        self.label_width_shift_pop.setText(_translate("Form", "Width shift", None))
        self.label_height_shift_pop.setToolTip(_translate("Form", tooltips["label_height_shift"], None))
        self.label_height_shift_pop.setText(_translate("Form", "Height shift", None))
        self.label_zoom_pop.setToolTip(_translate("Form", tooltips["label_zoom"], None))
        self.label_zoom_pop.setText(_translate("Form", "Zoom", None))
        self.label_shear_pop.setToolTip(_translate("Form",tooltips["label_shear"], None))
        self.label_shear_pop.setText(_translate("Form", "Shear", None))
        self.lineEdit_Rotation_pop.setToolTip(_translate("Form", tooltips["label_Rotation"], None))
        self.lineEdit_widthShift_pop.setToolTip(_translate("Form", tooltips["label_width_shift"], None))
        self.lineEdit_heightShift_pop.setToolTip(_translate("Form", tooltips["label_height_shift"], None))
        self.lineEdit_zoomRange_pop.setToolTip(_translate("Form", tooltips["label_zoom"], None))
        self.lineEdit_shearRange_pop.setToolTip(_translate("Form", tooltips["label_shear"], None))
        self.tabWidget_DefineModel_pop.setTabText(self.tabWidget_DefineModel_pop.indexOf(self.tab_KerasImgAug_pop), _translate("Form", "Affine img. augm.", None))
        self.label_RefreshAfterNrEpochs_pop.setToolTip(_translate("Form", tooltips["label_RefreshAfterNrEpochs"], None))
        self.label_RefreshAfterNrEpochs_pop.setText(_translate("Form", "Refresh after nr. epochs:", None))
        self.spinBox_RefreshAfterNrEpochs_pop.setToolTip(_translate("Form", tooltips["label_RefreshAfterNrEpochs"], None))
        self.groupBox_BrightnessAugmentation_pop.setToolTip(_translate("Form", tooltips["groupBox_BrightnessAugmentation"], None))
        self.groupBox_BrightnessAugmentation_pop.setTitle(_translate("Form", "Brightness augmentation", None))
        self.label_Plus_pop.setToolTip(_translate("Form", tooltips["spinBox_PlusLower"], None))
        self.label_Plus_pop.setText(_translate("Form", "Add.", None))
        self.spinBox_PlusLower_pop.setToolTip(_translate("Form", tooltips["spinBox_PlusLower"], None))
        #self.label_PlusTo_pop.setToolTip(_translate("Form", "<html><head/><body><p>Define upper threshold for additive offset</p></body></html>", None))
        #self.label_PlusTo_pop.setText(_translate("Form", "...", None))
        self.spinBox_PlusUpper_pop.setToolTip(_translate("Form", tooltips["spinBox_PlusUpper"], None))
        self.label_Mult_pop.setToolTip(_translate("Form", tooltips["doubleSpinBox_MultLower"], None))
        self.label_Mult_pop.setText(_translate("Form", "Mult.", None))
        self.doubleSpinBox_MultLower_pop.setToolTip(_translate("Form", tooltips["doubleSpinBox_MultLower"], None))
        #self.label_Rotation_MultTo_pop.setText(_translate("Form", "...", None))
        self.groupBox_GaussianNoise_pop.setToolTip(_translate("Form", tooltips["groupBox_GaussianNoise"], None))
        self.groupBox_GaussianNoise_pop.setTitle(_translate("Form", "Gaussian noise", None))
        self.label_GaussianNoiseMean_pop.setToolTip(_translate("Form", tooltips["label_GaussianNoiseMean"], None))
        self.label_GaussianNoiseMean_pop.setText(_translate("Form", "Mean", None))
        self.doubleSpinBox_GaussianNoiseMean_pop.setToolTip(_translate("Form", tooltips["label_GaussianNoiseMean"], None))
        self.label_GaussianNoiseScale_pop.setToolTip(_translate("Form", tooltips["label_GaussianNoiseScale"], None))
        self.label_GaussianNoiseScale_pop.setText(_translate("Form", "Scale", None))
        self.doubleSpinBox_GaussianNoiseScale_pop.setToolTip(_translate("Form",tooltips["label_GaussianNoiseScale"], None))

        self.groupBox_colorAugmentation_pop.setTitle(_translate("Form", "Color augm.", None))
        self.checkBox_contrast_pop.setText(_translate("Form", "Contrast", None))
        self.checkBox_contrast_pop.setToolTip(_translate("Form", tooltips["checkBox_contrast"], None))
        self.checkBox_saturation_pop.setText(_translate("Form", "Saturation", None))
        self.checkBox_saturation_pop.setToolTip(_translate("Form", tooltips["checkBox_saturation"], None))
        self.checkBox_hue_pop.setText(_translate("Form", "Hue", None))
        self.checkBox_hue_pop.setToolTip(_translate("Form",tooltips["checkBox_hue"], None))




        self.groupBox_blurringAug_pop.setTitle(_translate("MainWindow", "Blurring", None))
        self.groupBox_blurringAug_pop.setToolTip(_translate("MainWindow", tooltips["groupBox_blurringAug"], None))
        self.label_motionBlurKernel_pop.setToolTip(_translate("MainWindow", tooltips["label_motionBlurKernel"], None))
        self.label_motionBlurKernel_pop.setText(_translate("MainWindow", "Kernel", None))
        self.lineEdit_motionBlurAngle_pop.setToolTip(_translate("MainWindow", tooltips["lineEdit_motionBlurAngle"], None))
        #self.lineEdit_motionBlurAngle_pop.setText(_translate("MainWindow", "-10,10", None))
        self.label_avgBlurMin_pop.setToolTip(_translate("MainWindow", tooltips["label_avgBlurMin"], None))
        self.label_avgBlurMin_pop.setText(_translate("MainWindow", "Min", None))
        self.spinBox_gaussBlurMax_pop.setToolTip(_translate("MainWindow", tooltips["spinBox_gaussBlurMax"], None))
        self.checkBox_motionBlur_pop.setToolTip(_translate("MainWindow", tooltips["checkBox_motionBlur"], None))
        self.checkBox_motionBlur_pop.setText(_translate("MainWindow", "Motion", None))
        self.spinBox_avgBlurMin_pop.setToolTip(_translate("MainWindow", tooltips["spinBox_avgBlurMin"], None))
        self.spinBox_gaussBlurMin_pop.setToolTip(_translate("MainWindow", tooltips["spinBox_avgBlurMin"], None))
        self.label_motionBlurAngle_pop.setToolTip(_translate("MainWindow", tooltips["label_motionBlurAngle"], None))
        self.label_motionBlurAngle_pop.setText(_translate("MainWindow", "Angle", None))
        self.label_gaussBlurMin_pop.setToolTip(_translate("MainWindow", tooltips["label_gaussBlurMin"], None))
        self.label_gaussBlurMin_pop.setText(_translate("MainWindow", "Min", None))
        self.label_gaussBlurMin_pop.setToolTip(_translate("MainWindow", tooltips["label_gaussBlurMin"], None))
        self.checkBox_gaussBlur_pop.setToolTip(_translate("MainWindow", tooltips["checkBox_gaussBlur"], None))
        self.checkBox_gaussBlur_pop.setText(_translate("MainWindow", "Gauss", None))
        self.spinBox_avgBlurMax_pop.setToolTip(_translate("MainWindow", tooltips["spinBox_avgBlurMax"], None))
        self.label_gaussBlurMax_pop.setToolTip(_translate("MainWindow", tooltips["spinBox_avgBlurMax"], None))
        self.label_gaussBlurMax_pop.setText(_translate("MainWindow", "Max", None))
        self.label_gaussBlurMax_pop.setToolTip(_translate("MainWindow", tooltips["spinBox_avgBlurMax"], None))
        self.checkBox_avgBlur_pop.setToolTip(_translate("MainWindow", tooltips["checkBox_avgBlur"], None))        
        self.checkBox_avgBlur_pop.setText(_translate("MainWindow", "Average", None))
        
        self.label_avgBlurMax_pop.setToolTip(_translate("MainWindow", tooltips["label_avgBlurMax"], None))
        self.label_avgBlurMax_pop.setText(_translate("MainWindow", "Max", None))
        self.spinBox_avgBlurMin_pop.setToolTip(_translate("MainWindow", tooltips["spinBox_avgBlurMin"], None))
        self.spinBox_avgBlurMax_pop.setToolTip(_translate("MainWindow", tooltips["spinBox_avgBlurMax"], None))
        self.lineEdit_motionBlurKernel_pop.setToolTip(_translate("MainWindow", tooltips["lineEdit_motionBlurKernel"], None))
        #self.lineEdit_motionBlurKernel_pop.setText(_translate("MainWindow", "0,5", None))

        
        self.tabWidget_DefineModel_pop.setTabText(self.tabWidget_DefineModel_pop.indexOf(self.tab_BrightnessAug_pop), _translate("Form", "Brightn/Color augm.", None))
        self.label_ShowIndex_pop.setText(_translate("Form", "Class", None))
        self.pushButton_ShowExamleImgs_pop.setText(_translate("Form", "Show", None))
        self.tabWidget_DefineModel_pop.setTabText(self.tabWidget_DefineModel_pop.indexOf(self.tab_ExampleImgs_pop), _translate("Form", "Example imgs.", None))
        self.tabWidget_DefineModel_pop.setTabToolTip(self.tabWidget_DefineModel_pop.indexOf(self.tab_ExampleImgs_pop), _translate("Form", "<html><head/><body><p>Show random example images of the training data</p></body></html>", None))

        self.groupBox_expertMode_pop.setToolTip(_translate("Form", tooltips["groupBox_expertMode"], None))
        self.groupBox_expertMode_pop.setTitle(_translate("Form", "Expert mode", None))
        self.groupBox_modelKerasFit_pop.setTitle(_translate("Form", "In model_keras.fit()", None))       




        self.groupBox_lossOptimizer.setToolTip(_translate("MainWindow", tooltips["groupBox_lossOptimizer"], None))
        self.checkBox_optimizer_pop.setToolTip(_translate("MainWindow", tooltips["label_optimizer"] , None))
        self.comboBox_optimizer.setToolTip(_translate("MainWindow", tooltips["label_optimizer"] , None))    
        self.pushButton_optimizer_pop.setToolTip(_translate("MainWindow", "Show advanced options for optimizer", None))
        self.checkBox_expt_loss_pop.setToolTip(_translate("MainWindow", tooltips["label_expt_loss"], None))
        self.comboBox_expt_loss_pop.setToolTip(_translate("MainWindow", tooltips["label_expt_loss"], None))
        self.checkBox_lossW.setToolTip(_translate("MainWindow",tooltips["checkBox_lossW"] , None))
        self.lineEdit_lossW.setToolTip(_translate("MainWindow", tooltips["checkBox_lossW"], None))

        self.groupBox_learningRate_pop.setToolTip(_translate("Form", tooltips["groupBox_learningRate"], None))
        self.groupBox_learningRate_pop.setTitle(_translate("Form", "Learning Rate", None))
        self.doubleSpinBox_learningRate.setToolTip(_translate("Form", tooltips["doubleSpinBox_learningRate"], None))

        self.radioButton_LrConst.setToolTip(_translate("MainWindow", tooltips["doubleSpinBox_learningRate"],None))
        self.radioButton_LrCycl.setToolTip(_translate("MainWindow", tooltips["radioButton_LrCycl"],None))
        self.label_cycLrMethod.setToolTip(_translate("MainWindow", tooltips["comboBox_cycLrMethod"], None))    
        self.label_cycLrMin.setToolTip(_translate("MainWindow", tooltips["label_cycLrMin"],None))
        self.lineEdit_cycLrMin.setToolTip(_translate("MainWindow", tooltips["label_cycLrMin"],None))
        self.lineEdit_cycLrMax.setToolTip(_translate("MainWindow", tooltips["label_cycLrMax"],None))
        self.comboBox_cycLrMethod.setToolTip(_translate("MainWindow", tooltips["comboBox_cycLrMethod"],None))
        self.pushButton_cycLrPopup.setToolTip(_translate("MainWindow", tooltips["pushButton_cycLrPopup"],None))
        self.radioButton_LrExpo.setToolTip(_translate("MainWindow", tooltips["radioButton_LrExpo"],None))
        self.label_expDecInitLr.setToolTip(_translate("MainWindow", tooltips["label_expDecInitLr"],None))
        self.doubleSpinBox_expDecInitLr.setToolTip(_translate("MainWindow", tooltips["radioButton_LrExpo"],None))
        self.label_expDecSteps.setToolTip(_translate("MainWindow", tooltips["label_expDecSteps"],None))
        self.spinBox_expDecSteps.setToolTip(_translate("MainWindow", tooltips["label_expDecSteps"],None))
        self.label_expDecRate.setToolTip(_translate("MainWindow", tooltips["label_expDecRate"],None))
        self.doubleSpinBox_expDecRate.setToolTip(_translate("MainWindow", tooltips["label_expDecRate"],None))

        self.pushButton_LR_finder.setToolTip(_translate("MainWindow", tooltips["groupBox_LrSettings"], None))    
        self.pushButton_LR_plot.setToolTip(_translate("MainWindow", tooltips["pushButton_LR_plot"], None))



    














        self.checkBox_trainLastNOnly_pop.setToolTip(_translate("Form", tooltips["checkBox_trainLastNOnly"], None))
        self.checkBox_trainLastNOnly_pop.setText(_translate("Form", "Train only last N layers", None))
        self.spinBox_trainLastNOnly_pop.setToolTip(_translate("Form", tooltips["spinBox_trainLastNOnly"], None))
        self.checkBox_trainDenseOnly_pop.setToolTip(_translate("Form", tooltips["checkBox_trainDenseOnly"], None))
        self.checkBox_trainDenseOnly_pop.setText(_translate("Form", "Train Dense only", None))

        self.label_batchSize_pop.setToolTip(_translate("Form", tooltips["label_batchSize"], None))
        self.label_batchSize_pop.setText(_translate("Form", "Batch size", None))
        self.spinBox_batchSize.setToolTip(_translate("Form", tooltips["label_batchSize"], None))
        self.label_epochs_pop.setToolTip(_translate("Form", tooltips["label_epochs"], None))
        self.label_epochs_pop.setText(_translate("Form", "Epochs", None))
        self.spinBox_epochs.setToolTip(_translate("Form", tooltips["label_epochs"], None))



        self.groupBox_learningRate_pop.setTitle(_translate("Form", "Learning rate (LR)", None))
        self.groupBox_lossOptimizer.setTitle(_translate("MainWindow", "Loss / Optimizer", None))

        self.radioButton_LrConst.setText(_translate("Form", "Constant", None))
        self.label_LrConst_pop.setText(_translate("Form", "LR", None))
        self.radioButton_LrCycl.setText(_translate("Form", "Cyclical", None))
        self.label_cycLrMin.setText(_translate("Form", "Range", None))
        self.label_cycLrMethod.setText(_translate("Form", "Method", None))
        self.comboBox_cycLrMethod.setItemText(0, _translate("Form", "triangular", None))
        self.comboBox_cycLrMethod.setItemText(1, _translate("Form", "triangular2", None))
        self.comboBox_cycLrMethod.setItemText(2, _translate("Form", "exp_range", None))
        self.pushButton_cycLrPopup.setText(_translate("Form", "...", None))
        self.radioButton_LrExpo.setText(_translate("Form", "Expo.", None))
        self.label_expDecInitLr.setText(_translate("Form", "Initial LR", None))
        self.label_expDecSteps.setText(_translate("Form", "Decay steps", None))
        self.label_expDecRate.setText(_translate("Form", "Decay rate", None))
        self.pushButton_LR_finder.setText(_translate("Form", "LR Screen", None))
        self.pushButton_LR_plot.setText(_translate("Form", "Plot", None))

        self.groupBox_expt_imgProc_pop.setTitle(_translate("Form", "Image processing", None))
        self.label_paddingMode_pop.setText(_translate("Form", "Padding mode", None))
        self.comboBox_paddingMode_pop.setToolTip(_translate("Form", tooltips["label_paddingMode"], None))
        
        self.comboBox_paddingMode_pop.setItemText(0, _translate("Form", "constant", None))
        self.comboBox_paddingMode_pop.setItemText(1, _translate("Form", "edge", None))
        self.comboBox_paddingMode_pop.setItemText(2, _translate("Form", "reflect", None))
        self.comboBox_paddingMode_pop.setItemText(3, _translate("Form", "symmetric", None))
        self.comboBox_paddingMode_pop.setItemText(4, _translate("Form", "wrap", None))
        self.comboBox_paddingMode_pop.setItemText(5, _translate("Form", "delete", None))
        self.comboBox_paddingMode_pop.setItemText(6, _translate("Form", "alternate", None))

        self.comboBox_zoomOrder.setItemText(0, _translate("MainWindow", "nearest neighbor (cv2.INTER_NEAREST)", None))
        self.comboBox_zoomOrder.setItemText(1, _translate("MainWindow", "lin. interp. (cv2.INTER_LINEAR)", None))
        self.comboBox_zoomOrder.setItemText(2, _translate("MainWindow", "quadr. interp. (cv2.INTER_AREA)", None))
        self.comboBox_zoomOrder.setItemText(3, _translate("MainWindow", "cubic interp. (cv2.INTER_CUBIC)", None))
        self.comboBox_zoomOrder.setItemText(4, _translate("MainWindow", "Lanczos 4 (cv2.INTER_LANCZOS4)", None))
        
        zoomitems = [self.comboBox_zoomOrder.itemText(i) for i in range(self.comboBox_zoomOrder.count())]
        width=self.comboBox_zoomOrder.fontMetrics().boundingRect(max(zoomitems, key=len)).width()
        self.comboBox_zoomOrder.view().setFixedWidth(width+10)             

        self.groupBox_regularization_pop.setTitle(_translate("Form", "Regularization", None))
        self.checkBox_expt_loss_pop.setText(_translate("Form", "Loss", None))
        self.comboBox_expt_loss_pop.setItemText(0, _translate("Form", "categorical_crossentropy", None))
        #self.comboBox_expt_loss_pop.setItemText(1, _translate("Form", "sparse_categorical_crossentropy", None))
        self.comboBox_expt_loss_pop.setItemText(1, _translate("Form", "mean_squared_error", None))
        self.comboBox_expt_loss_pop.setItemText(2, _translate("Form", "mean_absolute_error", None))
        self.comboBox_expt_loss_pop.setItemText(3, _translate("Form", "mean_absolute_percentage_error", None))
        self.comboBox_expt_loss_pop.setItemText(4, _translate("Form", "mean_squared_logarithmic_error", None))
        self.comboBox_expt_loss_pop.setItemText(5, _translate("Form", "squared_hinge", None))
        self.comboBox_expt_loss_pop.setItemText(6, _translate("Form", "hinge", None))
        self.comboBox_expt_loss_pop.setItemText(7, _translate("Form", "categorical_hinge", None))
        self.comboBox_expt_loss_pop.setItemText(8, _translate("Form", "logcosh", None))
        #self.comboBox_expt_loss_pop.setItemText(10, _translate("Form", "huber_loss", None))
        self.comboBox_expt_loss_pop.setItemText(9, _translate("Form", "binary_crossentropy", None))
        self.comboBox_expt_loss_pop.setItemText(10, _translate("Form", "kullback_leibler_divergence", None))
        self.comboBox_expt_loss_pop.setItemText(11, _translate("Form", "poisson", None))
        self.comboBox_expt_loss_pop.setItemText(12, _translate("Form", "cosine_proximity", None))
        #self.comboBox_expt_loss_pop.setItemText(15, _translate("Form", "is_categorical_crossentropy", None))

        self.checkBox_lossW.setText(_translate("Form", "Loss Weights", None))
        self.checkBox_lossW.setToolTip(_translate("Form", tooltips["checkBox_lossW"], None))
        self.lineEdit_lossW.setToolTip(_translate("Form", tooltips["checkBox_lossW"], None))
        self.pushButton_lossW.setToolTip(_translate("Form", tooltips["checkBox_lossW"], None))
        self.pushButton_lossW.setText(_translate("Form", "...", None))
        self.pushButton_optimizer_pop.setText(_translate("MainWindow", "...", None))
        self.pushButton_optimizer_pop.setToolTip(_translate("MainWindow", "Show advanced options for optimizer", None))


        self.checkBox_optimizer_pop.setText(_translate("Form", "Optimizer", None))
        self.comboBox_optimizer.setItemText(0, _translate("Form", "Adam", None))
        self.comboBox_optimizer.setItemText(1, _translate("Form", "SGD", None))
        self.comboBox_optimizer.setItemText(2, _translate("Form", "RMSprop", None))
        self.comboBox_optimizer.setItemText(3, _translate("Form", "Adagrad", None))
        self.comboBox_optimizer.setItemText(4, _translate("Form", "Adadelta", None))
        self.comboBox_optimizer.setItemText(5, _translate("Form", "Adamax", None))
        self.comboBox_optimizer.setItemText(6, _translate("Form", "Nadam", None))


        self.checkBox_dropout_pop.setToolTip(_translate("Form", tooltips["checkBox_dropout"], None))
        self.checkBox_dropout_pop.setText(_translate("Form", "Change Dropout to", None))
        self.lineEdit_dropout_pop.setToolTip(_translate("Form", tooltips["checkBox_dropout"], None))

#        self.checkBox_pTr_pop.setText(_translate("Form", "Partial trainability", None))
#        self.checkBox_pTr_pop.setToolTip(_translate("Form", tooltips["checkBox_partialTrainability"], None))
#        self.lineEdit_pTr_pop.setToolTip(_translate("Form", tooltips["checkBox_partialTrainability"], None))
#        self.pushButton_pTr_pop.setText(_translate("Form", "...", None))
#        self.pushButton_pTr_pop.setToolTip(_translate("Form", tooltips["checkBox_partialTrainability"], None))


        self.tabWidget_DefineModel_pop.setTabText(self.tabWidget_DefineModel_pop.indexOf(self.tab_expertMode_pop), _translate("Form", "Expert", None))
        
    #Functions for Keras augmentation checkboxes
    def keras_changed_rotation_pop(self,on_or_off):
        if on_or_off==0:
            self.lineEdit_Rotation_pop.setText(str(0))
            self.lineEdit_Rotation_pop.setEnabled(False)
        elif on_or_off==2:
            self.lineEdit_Rotation_pop.setText(str(Default_dict ["rotation"]))
            self.lineEdit_Rotation_pop.setEnabled(True)
        else:
            return
    def keras_changed_width_shift_pop(self,on_or_off):
        if on_or_off==0:
            self.lineEdit_widthShift_pop.setText(str(0))
            self.lineEdit_widthShift_pop.setEnabled(False)
        elif on_or_off==2:
            self.lineEdit_widthShift_pop.setText(str(Default_dict ["width_shift"]))
            self.lineEdit_widthShift_pop.setEnabled(True)
        else:
            return
    def keras_changed_height_shift_pop(self,on_or_off):
        if on_or_off==0:
            self.lineEdit_heightShift_pop.setText(str(0))
            self.lineEdit_heightShift_pop.setEnabled(False)
        elif on_or_off==2:
            self.lineEdit_heightShift_pop.setText(str(Default_dict ["height_shift"]))
            self.lineEdit_heightShift_pop.setEnabled(True)
        else:
            return
    def keras_changed_zoom_pop(self,on_or_off):
        if on_or_off==0:
            self.lineEdit_zoomRange_pop.setText(str(0))
            self.lineEdit_zoomRange_pop.setEnabled(False)
        elif on_or_off==2:
            self.lineEdit_zoomRange_pop.setText(str(Default_dict ["zoom"]))
            self.lineEdit_zoomRange_pop.setEnabled(True)
        else:
            return
    def keras_changed_shear_pop(self,on_or_off):
        if on_or_off==0:
            self.lineEdit_shearRange_pop.setText(str(0))
            self.lineEdit_shearRange_pop.setEnabled(False)
        elif on_or_off==2:
            self.lineEdit_shearRange_pop.setText(str(Default_dict ["shear"]))
            self.lineEdit_shearRange_pop.setEnabled(True)
        else:
            return
    def keras_changed_brightplus_pop(self,on_or_off):
        if on_or_off==0:
            self.spinBox_PlusLower_pop.setValue(0)
            self.spinBox_PlusLower_pop.setEnabled(False)
            self.spinBox_PlusUpper_pop.setValue(0)
            self.spinBox_PlusUpper_pop.setEnabled(False)
        elif on_or_off==2:
            self.spinBox_PlusLower_pop.setValue(Default_dict ["Brightness add. lower"])
            self.spinBox_PlusLower_pop.setEnabled(True)
            self.spinBox_PlusUpper_pop.setValue(Default_dict ["Brightness add. upper"])
            self.spinBox_PlusUpper_pop.setEnabled(True)
        else:
            return
    def keras_changed_brightmult_pop(self,on_or_off):
        if on_or_off==0:
            self.doubleSpinBox_MultLower_pop.setValue(1.0)
            self.doubleSpinBox_MultLower_pop.setEnabled(False)
            self.doubleSpinBox_MultUpper_pop.setValue(1.0)
            self.doubleSpinBox_MultUpper_pop.setEnabled(False)
        elif on_or_off==2:
            self.doubleSpinBox_MultLower_pop.setValue(Default_dict ["Brightness mult. lower"])
            self.doubleSpinBox_MultLower_pop.setEnabled(True)
            self.doubleSpinBox_MultUpper_pop.setValue(Default_dict ["Brightness mult. upper"])
            self.doubleSpinBox_MultUpper_pop.setEnabled(True)
        else:
            return
    def keras_changed_noiseMean_pop(self,on_or_off):
        if on_or_off==0:
            self.doubleSpinBox_GaussianNoiseMean_pop.setValue(0.0)
            self.doubleSpinBox_GaussianNoiseMean_pop.setEnabled(False)
        elif on_or_off==2:
            self.doubleSpinBox_GaussianNoiseMean_pop.setValue(Default_dict ["Gaussnoise Mean"])
            self.doubleSpinBox_GaussianNoiseMean_pop.setEnabled(True)
        else:
            return
    def keras_changed_noiseScale_pop(self,on_or_off):
        if on_or_off==0:
            self.doubleSpinBox_GaussianNoiseScale_pop.setValue(0.0)
            self.doubleSpinBox_GaussianNoiseScale_pop.setEnabled(False)
        elif on_or_off==2:
            self.doubleSpinBox_GaussianNoiseScale_pop.setValue(Default_dict ["Gaussnoise Scale"])
            self.doubleSpinBox_GaussianNoiseScale_pop.setEnabled(True)
        else:
            return
    def keras_changed_contrast_pop(self,on_or_off):
        if on_or_off==0:
            self.doubleSpinBox_contrastLower_pop.setEnabled(False)
            self.doubleSpinBox_contrastHigher_pop.setEnabled(False)
        elif on_or_off==2:
            self.doubleSpinBox_contrastLower_pop.setEnabled(True)
            self.doubleSpinBox_contrastHigher_pop.setEnabled(True)
        else:
            return
    def keras_changed_saturation_pop(self,on_or_off):
        if on_or_off==0:
            self.doubleSpinBox_saturationLower_pop.setEnabled(False)
            self.doubleSpinBox_saturationHigher_pop.setEnabled(False)
        elif on_or_off==2:
            self.doubleSpinBox_saturationLower_pop.setEnabled(True)
            self.doubleSpinBox_saturationHigher_pop.setEnabled(True)
        else:
            return
    def keras_changed_hue_pop(self,on_or_off):
        if on_or_off==0:
            self.doubleSpinBox_hueDelta_pop.setEnabled(False)
        elif on_or_off==2:
            self.doubleSpinBox_hueDelta_pop.setEnabled(True)
        else:
            return


    def expert_mode_off_pop(self,on_or_off):
        """
        Reset all values on the expert tab to the default values, excluding the metrics
        metrics are defined only once when starting fitting and should not be changed
        """
        if on_or_off==0: #switch off
            self.spinBox_batchSize.setValue(Default_dict["spinBox_batchSize"])
            self.spinBox_epochs.setValue(1)
            self.checkBox_expt_loss_pop.setChecked(False)
            self.expert_loss_off_pop(0)
            self.groupBox_learningRate_pop.setChecked(False)        
            self.expert_learningrate_off_pop(0)
            self.checkBox_optimizer_pop.setChecked(False)
            self.expert_optimizer_off_pop(0)


    def expert_loss_off_pop(self,on_or_off):
        if on_or_off==0: #switch off
            #switch back to categorical_crossentropy 
            index = self.comboBox_expt_loss_pop.findText("categorical_crossentropy", QtCore.Qt.MatchFixedString)
            if index >= 0:
                self.comboBox_expt_loss_pop.setCurrentIndex(index)
        
    def expert_learningrate_off_pop(self,on_or_off):
        if on_or_off==0: #switch off
            #which optimizer is used? (there are different default learning-rates
            #for each optimizer!)
            optimizer = str(self.comboBox_optimizer.currentText())
            self.doubleSpinBox_learningRate.setValue(Default_dict["doubleSpinBox_learningRate_"+optimizer])
            self.doubleSpinBox_learningRate.setEnabled(False)
            self.radioButton_LrCycl.setChecked(False)
            self.radioButton_LrExpo.setChecked(False)
            self.radioButton_LrConst.setChecked(True)

    def expert_optimizer_off_pop(self,on_or_off):
        if on_or_off==0: #switch off, set back to categorical_crossentropy
            optimizer = "Adam"
            index = self.comboBox_optimizer.findText(optimizer, QtCore.Qt.MatchFixedString)
            if index >= 0:
                self.comboBox_optimizer.setCurrentIndex(index)
                #also reset the learning rate to the default
                self.doubleSpinBox_learningRate.setValue(Default_dict["doubleSpinBox_learningRate_"+optimizer])

#    def expert_optimizer_changed_pop(self,value):
#        #set the learning rate to the default for this optimizer
#        optimizer = value
#        value_current = float(self.doubleSpinBox_learningRate.value())
#        value_wanted = Default_dict["doubleSpinBox_learningRate_"+optimizer]
#        text = str(self.textBrowser_FittingInfo.toPlainText())
#        if value_current!=value_wanted and "Epoch" in text:#avoid that the message pops up when window is created
#            self.doubleSpinBox_learningRate.setValue(value_wanted)
#            self.doubleSpinBox_expDecInitLr.setValue(value_wanted)
#
#            #Inform user
#            msg = QtWidgets.QMessageBox()
#            msg.setIcon(QtWidgets.QMessageBox.Information)       
#            msg.setWindowTitle("Learning rate to default")
#            msg.setText("Learning rate was set to the default for "+optimizer)
#            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
#            msg.exec_()
#            return

    def partialtrainability_activated_pop(self,listindex):#same function like partialTrainability but on fitting popup
        print("Not implemented yet")
        print("Building site")



















class popup_trainability(QtWidgets.QWidget):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(558, 789)
        self.gridLayout_2 = QtWidgets.QGridLayout(Form)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
#        self.pushButton_pop_pTr_reset = QtWidgets.QPushButton(Form)
#        self.pushButton_pop_pTr_reset.setObjectName("pushButton_pop_pTr_reset")
#        self.horizontalLayout_8.addWidget(self.pushButton_pop_pTr_reset)
        spacerItem = QtWidgets.QSpacerItem(218, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacerItem)
        self.pushButton_pop_pTr_update = QtWidgets.QPushButton(Form)
        self.pushButton_pop_pTr_update.setObjectName("pushButton_pop_pTr_update")
        self.horizontalLayout_8.addWidget(self.pushButton_pop_pTr_update)
        self.pushButton_pop_pTr_ok = QtWidgets.QPushButton(Form)
        self.pushButton_pop_pTr_ok.setObjectName("pushButton_pop_pTr_ok")
        self.horizontalLayout_8.addWidget(self.pushButton_pop_pTr_ok)
        self.gridLayout_2.addLayout(self.horizontalLayout_8, 1, 0, 1, 1)
        self.splitter = QtWidgets.QSplitter(Form)
        self.splitter.setOrientation(QtCore.Qt.Vertical)
        self.splitter.setObjectName("splitter")
        self.groupBox = QtWidgets.QGroupBox(self.splitter)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout.setContentsMargins(-1, 5, -1, 5)
        self.gridLayout.setSpacing(3)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_pop_pTr_modelPath = QtWidgets.QLabel(self.groupBox)
        self.label_pop_pTr_modelPath.setObjectName("label_pop_pTr_modelPath")
        self.horizontalLayout_4.addWidget(self.label_pop_pTr_modelPath)
        self.lineEdit_pop_pTr_modelPath = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_pop_pTr_modelPath.setEnabled(False)
        self.lineEdit_pop_pTr_modelPath.setObjectName("lineEdit_pop_pTr_modelPath")
        self.horizontalLayout_4.addWidget(self.lineEdit_pop_pTr_modelPath)
        self.gridLayout.addLayout(self.horizontalLayout_4, 0, 0, 1, 3)
#        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
#        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
#        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
#        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
#        self.label_pop_pTr_arch = QtWidgets.QLabel(self.groupBox)
#        self.label_pop_pTr_arch.setObjectName("label_pop_pTr_arch")
#        self.horizontalLayout_5.addWidget(self.label_pop_pTr_arch)
#        self.lineEdit_pop_pTr_arch = QtWidgets.QLineEdit(self.groupBox)
#        self.lineEdit_pop_pTr_arch.setEnabled(False)
#        self.lineEdit_pop_pTr_arch.setObjectName("lineEdit_pop_pTr_arch")
#        self.horizontalLayout_5.addWidget(self.lineEdit_pop_pTr_arch)
#        self.horizontalLayout_7.addLayout(self.horizontalLayout_5)
#        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
#        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
#        self.label_pop_pTr_norm = QtWidgets.QLabel(self.groupBox)
#        self.label_pop_pTr_norm.setObjectName("label_pop_pTr_norm")
#        self.horizontalLayout_3.addWidget(self.label_pop_pTr_norm)
#        self.comboBox_pop_pTr_norm = QtWidgets.QComboBox(self.groupBox)
#        self.comboBox_pop_pTr_norm.setEnabled(False)
#        self.comboBox_pop_pTr_norm.setObjectName("comboBox_pop_pTr_norm")
#        self.horizontalLayout_3.addWidget(self.comboBox_pop_pTr_norm)
#        self.horizontalLayout_7.addLayout(self.horizontalLayout_3)
#        self.gridLayout.addLayout(self.horizontalLayout_7, 1, 0, 1, 3)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_pop_pTr_inpSize = QtWidgets.QLabel(self.groupBox)
        self.label_pop_pTr_inpSize.setObjectName("label_pop_pTr_inpSize")
        self.horizontalLayout.addWidget(self.label_pop_pTr_inpSize)
        self.spinBox_pop_pTr_inpSize = QtWidgets.QSpinBox(self.groupBox)
        self.spinBox_pop_pTr_inpSize.setEnabled(False)
        self.spinBox_pop_pTr_inpSize.setAccessibleName("")
        self.spinBox_pop_pTr_inpSize.setObjectName("spinBox_pop_pTr_inpSize")
        self.horizontalLayout.addWidget(self.spinBox_pop_pTr_inpSize)
        self.gridLayout.addLayout(self.horizontalLayout, 2, 0, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_pop_pTr_outpSize = QtWidgets.QLabel(self.groupBox)
        self.label_pop_pTr_outpSize.setObjectName("label_pop_pTr_outpSize")
        self.horizontalLayout_2.addWidget(self.label_pop_pTr_outpSize)
        self.spinBox_pop_pTr_outpSize = QtWidgets.QSpinBox(self.groupBox)
        self.spinBox_pop_pTr_outpSize.setEnabled(False)
        self.spinBox_pop_pTr_outpSize.setObjectName("spinBox_pop_pTr_outpSize")
        self.horizontalLayout_2.addWidget(self.spinBox_pop_pTr_outpSize)
        self.gridLayout.addLayout(self.horizontalLayout_2, 2, 1, 1, 1)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_pop_pTr_colorMode = QtWidgets.QLabel(self.groupBox)
        self.label_pop_pTr_colorMode.setObjectName("label_pop_pTr_colorMode")
        self.horizontalLayout_6.addWidget(self.label_pop_pTr_colorMode)
        self.comboBox_pop_pTr_colorMode = QtWidgets.QComboBox(self.groupBox)
        self.comboBox_pop_pTr_colorMode.setEnabled(False)
        self.comboBox_pop_pTr_colorMode.setObjectName("comboBox_pop_pTr_colorMode")
        self.horizontalLayout_6.addWidget(self.comboBox_pop_pTr_colorMode)
        self.gridLayout.addLayout(self.horizontalLayout_6, 2, 2, 1, 1)
        self.groupBox_pop_pTr_layersTable = QtWidgets.QGroupBox(self.splitter)
        self.groupBox_pop_pTr_layersTable.setObjectName("groupBox_pop_pTr_layersTable")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox_pop_pTr_layersTable)
        self.gridLayout_3.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.gridLayout_3.setContentsMargins(-1, 5, -1, 5)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.tableWidget_pop_pTr_layersTable = MyTable(0,5,self.groupBox_pop_pTr_layersTable)
        self.tableWidget_pop_pTr_layersTable.setObjectName("tableWidget_pop_pTr_layersTable")

        header_labels = ["Name", "Type" ,"No.Params", "No.Units", "Trainability"]
        self.tableWidget_pop_pTr_layersTable.setHorizontalHeaderLabels(header_labels) 
        header = self.tableWidget_pop_pTr_layersTable.horizontalHeader()
        for i in range(len(header_labels)):
            header.setResizeMode(i, QtWidgets.QHeaderView.ResizeToContents)        
        self.tableWidget_pop_pTr_layersTable.setAcceptDrops(True)
        self.tableWidget_pop_pTr_layersTable.setDragEnabled(True)
        self.tableWidget_pop_pTr_layersTable.resizeRowsToContents()

        self.gridLayout_3.addWidget(self.tableWidget_pop_pTr_layersTable, 0, 0, 1, 1)
        self.groupBox_pop_pTr_modelSummary = QtWidgets.QGroupBox(self.splitter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_pop_pTr_modelSummary.sizePolicy().hasHeightForWidth())
        self.groupBox_pop_pTr_modelSummary.setSizePolicy(sizePolicy)
        self.groupBox_pop_pTr_modelSummary.setBaseSize(QtCore.QSize(0, 0))
        self.groupBox_pop_pTr_modelSummary.setFlat(False)
        self.groupBox_pop_pTr_modelSummary.setCheckable(False)
        self.groupBox_pop_pTr_modelSummary.setObjectName("groupBox_pop_pTr_modelSummary")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.groupBox_pop_pTr_modelSummary)
        self.gridLayout_4.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.gridLayout_4.setContentsMargins(-1, 5, -1, 5)
        self.gridLayout_4.setHorizontalSpacing(7)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.textBrowser_pop_pTr_modelSummary = QtWidgets.QTextBrowser(self.groupBox_pop_pTr_modelSummary)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.textBrowser_pop_pTr_modelSummary.sizePolicy().hasHeightForWidth())
        self.textBrowser_pop_pTr_modelSummary.setSizePolicy(sizePolicy)
        self.textBrowser_pop_pTr_modelSummary.setMinimumSize(QtCore.QSize(0, 0))
        self.textBrowser_pop_pTr_modelSummary.setBaseSize(QtCore.QSize(0, 0))
        self.textBrowser_pop_pTr_modelSummary.setAutoFillBackground(False)
        self.textBrowser_pop_pTr_modelSummary.setObjectName("textBrowser_pop_pTr_modelSummary")
        self.gridLayout_4.addWidget(self.textBrowser_pop_pTr_modelSummary, 0, 0, 1, 1)
        self.gridLayout_2.addWidget(self.splitter, 0, 0, 1, 1)

        self.retranslateUi(Form)
                
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Partial trainability", None))
        self.tableWidget_pop_pTr_layersTable.setToolTip(_translate("Form", tooltips["tableWidget_pop_pTr_layersTable"], None))
        #self.pushButton_pop_pTr_reset.setText(_translate("Form", "Reset", None))
        #self.pushButton_pop_pTr_reset.setToolTip(_translate("Form", "<html><head/><body><p>Not implemented yet.</p></body></html>", None))
        self.pushButton_pop_pTr_update.setText(_translate("Form", "Update", None))
        self.pushButton_pop_pTr_update.setToolTip(_translate("Form", tooltips["pushButton_pop_pTr_update"], None))
        self.pushButton_pop_pTr_ok.setText(_translate("Form", "OK", None))
        self.pushButton_pop_pTr_ok.setToolTip(_translate("Form", tooltips["pushButton_pop_pTr_ok"], None))
        self.groupBox.setTitle(_translate("Form", "Model information", None))
        self.label_pop_pTr_modelPath.setText(_translate("Form", "Model path", None))
#        self.label_pop_pTr_arch.setText(_translate("Form", "Architecture", None))
#        self.label_pop_pTr_norm.setText(_translate("Form", "Normalization", None))
        self.label_pop_pTr_inpSize.setText(_translate("Form", "Input size", None))
        self.label_pop_pTr_outpSize.setText(_translate("Form", "Output classes", None))
        self.label_pop_pTr_colorMode.setText(_translate("Form", "Color Mode", None))
        self.groupBox_pop_pTr_layersTable.setTitle(_translate("Form", "Layers", None))
        self.groupBox_pop_pTr_modelSummary.setTitle(_translate("Form", "Model summary", None))


class popup_lossweights(QtWidgets.QWidget):
    def setupUi(self, Form_lossW):
        Form_lossW.setObjectName("Form_lossW")
        Form_lossW.resize(470, 310)
        self.gridLayout_2 = QtWidgets.QGridLayout(Form_lossW)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.groupBox_lossW = QtWidgets.QGroupBox(Form_lossW)
        self.groupBox_lossW.setObjectName("groupBox_lossW")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox_lossW)
        self.gridLayout.setObjectName("gridLayout")
        self.tableWidget_lossW = MyTable(0,5,self.groupBox_lossW)
        self.tableWidget_lossW.setObjectName("tableWidget_lossW")

        header_labels = ["Class", "Events tot." ,"Events/Epoch", "Events/Epoch[%]", "Loss weight"]
        self.tableWidget_lossW.setHorizontalHeaderLabels(header_labels) 
        header = self.tableWidget_lossW.horizontalHeader()
        for i in range(len(header_labels)):
            header.setResizeMode(i, QtWidgets.QHeaderView.ResizeToContents)        
        self.tableWidget_lossW.setAcceptDrops(True)
        self.tableWidget_lossW.setDragEnabled(True)
        self.tableWidget_lossW.resizeRowsToContents()

        self.gridLayout.addWidget(self.tableWidget_lossW, 0, 0, 1, 1)
        self.gridLayout_2.addWidget(self.groupBox_lossW, 0, 0, 1, 1)
        self.horizontalLayout_lossW_buttons = QtWidgets.QHBoxLayout()
        self.horizontalLayout_lossW_buttons.setObjectName("horizontalLayout_lossW_buttons")
#        self.pushButton_pop_lossW_reset = QtWidgets.QPushButton(Form_lossW)
#        self.pushButton_pop_lossW_reset.setObjectName("pushButton_pop_lossW_reset")
#        self.horizontalLayout_lossW_buttons.addWidget(self.pushButton_pop_lossW_reset)
        spacerItem = QtWidgets.QSpacerItem(218, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_lossW_buttons.addItem(spacerItem)
        self.pushButton_pop_lossW_cancel = QtWidgets.QPushButton(Form_lossW)
        self.pushButton_pop_lossW_cancel.setObjectName("pushButton_pop_lossW_cancel")
        self.horizontalLayout_lossW_buttons.addWidget(self.pushButton_pop_lossW_cancel)
        self.comboBox_lossW = QtWidgets.QComboBox(Form_lossW)
        self.comboBox_lossW.setObjectName("comboBox_lossW")
        self.comboBox_lossW.addItems(["None","Balanced","Custom"])
        self.horizontalLayout_lossW_buttons.addWidget(self.comboBox_lossW)
        self.pushButton_pop_lossW_ok = QtWidgets.QPushButton(Form_lossW)
        self.pushButton_pop_lossW_ok.setObjectName("pushButton_pop_lossW_ok")
        self.horizontalLayout_lossW_buttons.addWidget(self.pushButton_pop_lossW_ok)
        self.gridLayout_2.addLayout(self.horizontalLayout_lossW_buttons, 1, 0, 1, 1)

        self.retranslateUi(Form_lossW)
        QtCore.QMetaObject.connectSlotsByName(Form_lossW)


    def retranslateUi(self, Form_lossW):
        _translate = QtCore.QCoreApplication.translate
        Form_lossW.setWindowTitle(_translate("Form_lossW", "Custom loss weights per class", None))
        self.groupBox_lossW.setTitle(_translate("Form_lossW", "Training data - custom class weights", None))
        #self.pushButton_pop_lossW_reset.setText(_translate("Form_lossW", "Reset", None))
        self.pushButton_pop_lossW_cancel.setText(_translate("Form_lossW", "Cancel", None))
        self.pushButton_pop_lossW_ok.setText(_translate("Form_lossW", "OK", None))




class popup_imageLoadResize(QtWidgets.QWidget):
    def setupUi(self, Form_imageResize):
        Form_imageResize.setObjectName("Form_imageResize")
        Form_imageResize.resize(468, 270)
        self.gridLayout_3 = QtWidgets.QGridLayout(Form_imageResize)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.scrollArea_imgResize_occurences = QtWidgets.QScrollArea(Form_imageResize)
        self.scrollArea_imgResize_occurences.setWidgetResizable(True)
        self.scrollArea_imgResize_occurences.setObjectName("scrollArea_imgResize_occurences")
        self.scrollAreaWidgetContents_imgResize = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_imgResize.setGeometry(QtCore.QRect(0, 0, 423, 109))
        self.scrollAreaWidgetContents_imgResize.setObjectName("scrollAreaWidgetContents_imgResize")
        self.gridLayout = QtWidgets.QGridLayout(self.scrollAreaWidgetContents_imgResize)
        self.gridLayout.setObjectName("gridLayout")
        self.textBrowser_imgResize_occurences = QtWidgets.QTextBrowser(self.scrollAreaWidgetContents_imgResize)
        self.textBrowser_imgResize_occurences.setObjectName("textBrowser_imgResize_occurences")
        self.gridLayout.addWidget(self.textBrowser_imgResize_occurences, 0, 0, 1, 1)
        self.scrollArea_imgResize_occurences.setWidget(self.scrollAreaWidgetContents_imgResize)
        self.gridLayout_3.addWidget(self.scrollArea_imgResize_occurences, 2, 0, 1, 1)
        self.gridLayout_imageResizeOptions = QtWidgets.QGridLayout()
        self.gridLayout_imageResizeOptions.setObjectName("gridLayout_imageResizeOptions")
        self.label_imgResize_x_3 = QtWidgets.QLabel(Form_imageResize)
        self.label_imgResize_x_3.setObjectName("label_imgResize_x_3")
        self.gridLayout_imageResizeOptions.addWidget(self.label_imgResize_x_3, 2, 3, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(88, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_imageResizeOptions.addItem(spacerItem, 1, 5, 1, 1)
        self.label_imgResize_height = QtWidgets.QLabel(Form_imageResize)
        self.label_imgResize_height.setAlignment(QtCore.Qt.AlignCenter)
        self.label_imgResize_height.setObjectName("label_imgResize_height")
        self.gridLayout_imageResizeOptions.addWidget(self.label_imgResize_height, 0, 1, 1, 2)
        self.spinBox_ingResize_w_2 = QtWidgets.QSpinBox(Form_imageResize)
        self.spinBox_ingResize_w_2.setEnabled(False)
        self.spinBox_ingResize_w_2.setMaximum(999999)
        self.spinBox_ingResize_w_2.setObjectName("spinBox_ingResize_w_2")
        self.gridLayout_imageResizeOptions.addWidget(self.spinBox_ingResize_w_2, 2, 4, 1, 1)
        self.spinBox_ingResize_h_1 = QtWidgets.QSpinBox(Form_imageResize)
        self.spinBox_ingResize_h_1.setEnabled(False)
        self.spinBox_ingResize_h_1.setMaximum(999999)
        self.spinBox_ingResize_h_1.setObjectName("spinBox_ingResize_h_1")
        self.gridLayout_imageResizeOptions.addWidget(self.spinBox_ingResize_h_1, 1, 1, 1, 2)
        spacerItem1 = QtWidgets.QSpacerItem(88, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_imageResizeOptions.addItem(spacerItem1, 0, 5, 1, 1)
        self.pushButton_imgResize_ok = QtWidgets.QPushButton(Form_imageResize)
        self.pushButton_imgResize_ok.setEnabled(False)
        self.pushButton_imgResize_ok.setObjectName("pushButton_imgResize_ok")
        self.gridLayout_imageResizeOptions.addWidget(self.pushButton_imgResize_ok, 3, 5, 1, 1)
        self.spinBox_ingResize_w_1 = QtWidgets.QSpinBox(Form_imageResize)
        self.spinBox_ingResize_w_1.setEnabled(False)
        self.spinBox_ingResize_w_1.setMaximum(999999)
        self.spinBox_ingResize_w_1.setObjectName("spinBox_ingResize_w_1")
        self.gridLayout_imageResizeOptions.addWidget(self.spinBox_ingResize_w_1, 1, 4, 1, 1)
        self.comboBox_resizeMethod = QtWidgets.QComboBox(Form_imageResize)
        self.comboBox_resizeMethod.setEnabled(False)
        self.comboBox_resizeMethod.setObjectName("comboBox_resizeMethod")
        self.comboBox_resizeMethod.addItem("")
        self.comboBox_resizeMethod.addItem("")
        self.comboBox_resizeMethod.addItem("")
        self.comboBox_resizeMethod.addItem("")
        self.comboBox_resizeMethod.addItem("")
        self.gridLayout_imageResizeOptions.addWidget(self.comboBox_resizeMethod, 2, 5, 1, 1)
        self.pushButton_imgResize_cancel = QtWidgets.QPushButton(Form_imageResize)
        self.pushButton_imgResize_cancel.setObjectName("pushButton_imgResize_cancel")
        self.gridLayout_imageResizeOptions.addWidget(self.pushButton_imgResize_cancel, 3, 2, 1, 3)
        self.radioButton_imgResize_cropPad = QtWidgets.QRadioButton(Form_imageResize)
        self.radioButton_imgResize_cropPad.setObjectName("radioButton_imgResize_cropPad")
        self.gridLayout_imageResizeOptions.addWidget(self.radioButton_imgResize_cropPad, 1, 0, 1, 1)
        self.label_imgResize_x_2 = QtWidgets.QLabel(Form_imageResize)
        self.label_imgResize_x_2.setObjectName("label_imgResize_x_2")
        self.gridLayout_imageResizeOptions.addWidget(self.label_imgResize_x_2, 1, 3, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(148, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_imageResizeOptions.addItem(spacerItem2, 3, 0, 1, 2)
        self.spinBox_ingResize_h_2 = QtWidgets.QSpinBox(Form_imageResize)
        self.spinBox_ingResize_h_2.setEnabled(False)
        self.spinBox_ingResize_h_2.setMaximum(999999)
        self.spinBox_ingResize_h_2.setObjectName("spinBox_ingResize_h_2")
        self.gridLayout_imageResizeOptions.addWidget(self.spinBox_ingResize_h_2, 2, 1, 1, 2)
        self.label_imgResize_width = QtWidgets.QLabel(Form_imageResize)
        self.label_imgResize_width.setAlignment(QtCore.Qt.AlignCenter)
        self.label_imgResize_width.setObjectName("label_imgResize_width")
        self.gridLayout_imageResizeOptions.addWidget(self.label_imgResize_width, 0, 4, 1, 1)
        self.label_imgResize_method = QtWidgets.QLabel(Form_imageResize)
        self.label_imgResize_method.setObjectName("label_imgResize_method")
        self.gridLayout_imageResizeOptions.addWidget(self.label_imgResize_method, 0, 0, 1, 1)
        self.label_imgResize_x_1 = QtWidgets.QLabel(Form_imageResize)
        self.label_imgResize_x_1.setObjectName("label_imgResize_x_1")
        self.gridLayout_imageResizeOptions.addWidget(self.label_imgResize_x_1, 0, 3, 1, 1)
        self.radioButton_imgResize_interpolate = QtWidgets.QRadioButton(Form_imageResize)
        self.radioButton_imgResize_interpolate.setObjectName("radioButton_imgResize_interpolate")
        self.gridLayout_imageResizeOptions.addWidget(self.radioButton_imgResize_interpolate, 2, 0, 1, 1)
        self.gridLayout_3.addLayout(self.gridLayout_imageResizeOptions, 1, 0, 1, 1)
        self.label_imgResize_info = QtWidgets.QLabel(Form_imageResize)
        self.label_imgResize_info.setObjectName("label_imgResize_info")
        self.gridLayout_3.addWidget(self.label_imgResize_info, 0, 0, 1, 1)

        self.retranslateUi(Form_imageResize)
        self.radioButton_imgResize_cropPad.toggled['bool'].connect(self.spinBox_ingResize_h_1.setEnabled)
        #self.radioButton_imgResize_cropPad.toggled['bool'].connect(self.spinBox_ingResize_w_1.setEnabled)
        self.radioButton_imgResize_interpolate.toggled['bool'].connect(self.spinBox_ingResize_h_2.setEnabled)
        #self.radioButton_imgResize_interpolate.toggled['bool'].connect(self.spinBox_ingResize_w_2.setEnabled)
        self.radioButton_imgResize_interpolate.toggled['bool'].connect(self.comboBox_resizeMethod.setEnabled)
        self.radioButton_imgResize_cropPad.toggled['bool'].connect(self.pushButton_imgResize_ok.setEnabled)
        self.radioButton_imgResize_interpolate.toggled['bool'].connect(self.pushButton_imgResize_ok.setEnabled)
        self.spinBox_ingResize_h_1.valueChanged['int'].connect(self.spinBox_ingResize_h_2.setValue)
        self.spinBox_ingResize_h_1.valueChanged['int'].connect(self.spinBox_ingResize_w_1.setValue)
        self.spinBox_ingResize_h_1.valueChanged['int'].connect(self.spinBox_ingResize_w_2.setValue)
        self.spinBox_ingResize_h_2.valueChanged['int'].connect(self.spinBox_ingResize_w_1.setValue)
        self.spinBox_ingResize_h_2.valueChanged['int'].connect(self.spinBox_ingResize_w_2.setValue)
        self.spinBox_ingResize_h_2.valueChanged['int'].connect(self.spinBox_ingResize_h_1.setValue)

        QtCore.QMetaObject.connectSlotsByName(Form_imageResize)

    def retranslateUi(self, Form_imageResize):
        _translate = QtCore.QCoreApplication.translate
        Form_imageResize.setWindowTitle(_translate("Form_imageResize", "Import assistant for unequally sized images"))
        self.label_imgResize_x_3.setText(_translate("Form_imageResize", "x"))
        self.label_imgResize_height.setText(_translate("Form_imageResize", "Height"))
        self.pushButton_imgResize_ok.setText(_translate("Form_imageResize", "OK"))
        self.comboBox_resizeMethod.setItemText(0, _translate("Form_imageResize", "Nearest"))
        self.comboBox_resizeMethod.setItemText(1, _translate("Form_imageResize", "Linear"))
        self.comboBox_resizeMethod.setItemText(2, _translate("Form_imageResize", "Area"))
        self.comboBox_resizeMethod.setItemText(3, _translate("Form_imageResize", "Cubic"))
        self.comboBox_resizeMethod.setItemText(4, _translate("Form_imageResize", "Lanczos"))
        self.pushButton_imgResize_cancel.setText(_translate("Form_imageResize", "Cancel"))
        self.radioButton_imgResize_cropPad.setToolTip(_translate("Form_imageResize", "Images are resized by center cropping and/or padding."))
        self.radioButton_imgResize_cropPad.setText(_translate("Form_imageResize", "Crop/pad"))
        self.label_imgResize_x_2.setText(_translate("Form_imageResize", "x"))
        self.label_imgResize_width.setText(_translate("Form_imageResize", "Width"))
        self.label_imgResize_method.setText(_translate("Form_imageResize", "Method"))
        self.label_imgResize_x_1.setText(_translate("Form_imageResize", "x"))
        self.radioButton_imgResize_interpolate.setToolTip(_translate("Form_imageResize", "Images are resized by interpolation"))
        self.radioButton_imgResize_interpolate.setText(_translate("Form_imageResize", "Resize (interp.)"))
        self.label_imgResize_info.setText(_translate("Form_imageResize", "Detected unequal image sizes. Select a method to equalize image sizes:"))



class popup_cm_interaction(QtWidgets.QWidget):
    def setupUi(self, Form_cm_interaction):
        Form_cm_interaction.setObjectName("Form_cm_interaction")
        Form_cm_interaction.resize(702, 572)
        self.gridLayout_6 = QtWidgets.QGridLayout(Form_cm_interaction)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.groupBox_model = QtWidgets.QGroupBox(Form_cm_interaction)
        self.groupBox_model.setObjectName("groupBox_model")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.groupBox_model)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.lineEdit_loadModel = QtWidgets.QLineEdit(self.groupBox_model)
        self.lineEdit_loadModel.setEnabled(False)
        self.lineEdit_loadModel.setObjectName("lineEdit_loadModel")
        self.gridLayout_5.addWidget(self.lineEdit_loadModel, 0, 0, 1, 4)
        self.pushButton_showSummary = QtWidgets.QPushButton(self.groupBox_model)
        self.pushButton_showSummary.setObjectName("pushButton_showSummary")
        self.gridLayout_5.addWidget(self.pushButton_showSummary, 0, 4, 1, 1)
        self.label_inpImgSize = QtWidgets.QLabel(self.groupBox_model)
        self.label_inpImgSize.setObjectName("label_inpImgSize")
        self.gridLayout_5.addWidget(self.label_inpImgSize, 1, 0, 1, 1)
        self.spinBox_Crop_inpImgSize = QtWidgets.QSpinBox(self.groupBox_model)
        self.spinBox_Crop_inpImgSize.setEnabled(False)
        self.spinBox_Crop_inpImgSize.setObjectName("spinBox_Crop_inpImgSize")
        self.gridLayout_5.addWidget(self.spinBox_Crop_inpImgSize, 1, 1, 1, 1)
        self.label_outpSize = QtWidgets.QLabel(self.groupBox_model)
        self.label_outpSize.setObjectName("label_outpSize")
        self.gridLayout_5.addWidget(self.label_outpSize, 1, 2, 1, 1)
        self.spinBox_outpSize = QtWidgets.QSpinBox(self.groupBox_model)
        self.spinBox_outpSize.setEnabled(False)
        self.spinBox_outpSize.setObjectName("spinBox_outpSize")
        self.gridLayout_5.addWidget(self.spinBox_outpSize, 1, 3, 1, 1)
        self.pushButton_toTensorB = QtWidgets.QPushButton(self.groupBox_model)
        self.pushButton_toTensorB.setObjectName("pushButton_toTensorB")
        self.gridLayout_5.addWidget(self.pushButton_toTensorB, 1, 4, 1, 1)
        self.gridLayout_6.addWidget(self.groupBox_model, 0, 0, 1, 2)
        self.groupBox_imageShow = QtWidgets.QGroupBox(Form_cm_interaction)
        self.groupBox_imageShow.setObjectName("groupBox_imageShow")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox_imageShow)
        self.gridLayout.setObjectName("gridLayout")
        
        self.widget_image = pg.ImageView(self.groupBox_imageShow)
        self.widget_image.show()
        self.widget_image.setMinimumSize(QtCore.QSize(400, 400))
        #self.widget_image.setMaximumSize(QtCore.QSize(16777215, 91))
#        self.widget_image.ui.histogram.hide()
#        self.widget_image.ui.roiBtn.hide()
#        self.widget_image.ui.menuBtn.hide()
        self.widget_image.setObjectName("widget_image")
        
        self.gridLayout.addWidget(self.widget_image, 0, 0, 1, 1)
        self.gridLayout_6.addWidget(self.groupBox_imageShow, 1, 0, 2, 1)
        self.scrollArea_settings = QtWidgets.QScrollArea(Form_cm_interaction)
        self.scrollArea_settings.setWidgetResizable(True)
        self.scrollArea_settings.setObjectName("scrollArea_settings")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 247, 431))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.groupBox_image_Settings = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
        self.groupBox_image_Settings.setCheckable(True)
        self.groupBox_image_Settings.toggled.connect(self.image_on_off)
        
        self.groupBox_image_Settings.setObjectName("groupBox_image_Settings")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox_image_Settings)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_image_alpha = QtWidgets.QLabel(self.groupBox_image_Settings)
        self.label_image_alpha.setObjectName("label_image_alpha")
        self.gridLayout_2.addWidget(self.label_image_alpha, 0, 0, 1, 1)
        self.doubleSpinBox_image_alpha = QtWidgets.QDoubleSpinBox(self.groupBox_image_Settings)

        self.doubleSpinBox_image_alpha.setMinimum(0.0)
        self.doubleSpinBox_image_alpha.setMaximum(1.0)
        self.doubleSpinBox_image_alpha.setSingleStep(0.1)
        self.doubleSpinBox_image_alpha.setDecimals(3)
        self.doubleSpinBox_image_alpha.setValue(1.0)

        self.doubleSpinBox_image_alpha.setObjectName("doubleSpinBox_image_alpha")
        self.gridLayout_2.addWidget(self.doubleSpinBox_image_alpha, 0, 1, 1, 1)
        self.gridLayout_4.addWidget(self.groupBox_image_Settings, 0, 0, 1, 2)
        self.groupBox_gradCAM_Settings = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
        self.groupBox_gradCAM_Settings.setCheckable(True)
        self.groupBox_gradCAM_Settings.setChecked(False)
        self.groupBox_gradCAM_Settings.toggled.connect(self.gradCAM_on_off)
        
        self.groupBox_gradCAM_Settings.setObjectName("groupBox_gradCAM_Settings")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox_gradCAM_Settings)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_gradCAM_targetClass = QtWidgets.QLabel(self.groupBox_gradCAM_Settings)
        self.label_gradCAM_targetClass.setObjectName("label_gradCAM_targetClass")
        self.gridLayout_3.addWidget(self.label_gradCAM_targetClass, 0, 0, 1, 1)
        self.spinBox_gradCAM_targetClass = QtWidgets.QSpinBox(self.groupBox_gradCAM_Settings)
        
        self.spinBox_gradCAM_targetClass.setMinimum(0)
        self.spinBox_gradCAM_targetClass.setValue(0)
        
        self.spinBox_gradCAM_targetClass.setObjectName("spinBox_gradCAM_targetClass")
        self.gridLayout_3.addWidget(self.spinBox_gradCAM_targetClass, 0, 1, 1, 1)
        self.label_gradCAM_targetLayer = QtWidgets.QLabel(self.groupBox_gradCAM_Settings)
        self.label_gradCAM_targetLayer.setObjectName("label_gradCAM_targetLayer")
        self.gridLayout_3.addWidget(self.label_gradCAM_targetLayer, 1, 0, 1, 1)
        self.comboBox_gradCAM_targetLayer = QtWidgets.QComboBox(self.groupBox_gradCAM_Settings)
        self.comboBox_gradCAM_targetLayer.setObjectName("comboBox_gradCAM_targetLayer")
        self.gridLayout_3.addWidget(self.comboBox_gradCAM_targetLayer, 1, 1, 1, 1)
        self.label_gradCAM_colorMap = QtWidgets.QLabel(self.groupBox_gradCAM_Settings)
        self.label_gradCAM_colorMap.setObjectName("label_gradCAM_colorMap")
        self.gridLayout_3.addWidget(self.label_gradCAM_colorMap, 2, 0, 1, 1)
        self.comboBox_gradCAM_colorMap = QtWidgets.QComboBox(self.groupBox_gradCAM_Settings)
        cmaps = dir(cv2)
        ind = ["COLORMAP" in a for a in cmaps]
        cmaps = list(np.array(cmaps)[ind])
        cmaps = [a.split("_")[1] for a in cmaps]
        self.comboBox_gradCAM_colorMap.addItems(cmaps)
        #find "VIRIDIS" in cmaps
        ind = np.where(np.array(cmaps)=="VIRIDIS")[0][0]
        self.comboBox_gradCAM_colorMap.setCurrentIndex(ind)



        self.comboBox_gradCAM_colorMap.setObjectName("comboBox_gradCAM_colorMap")
        self.gridLayout_3.addWidget(self.comboBox_gradCAM_colorMap, 2, 1, 1, 1)
        self.label_gradCAM_alpha = QtWidgets.QLabel(self.groupBox_gradCAM_Settings)
        self.label_gradCAM_alpha.setObjectName("label_gradCAM_alpha")
        self.gridLayout_3.addWidget(self.label_gradCAM_alpha, 3, 0, 1, 1)
        self.doubleSpinBox_gradCAM_alpha = QtWidgets.QDoubleSpinBox(self.groupBox_gradCAM_Settings)
        
        self.doubleSpinBox_gradCAM_alpha.setMinimum(0.0)
        self.doubleSpinBox_gradCAM_alpha.setMaximum(1.0)
        self.doubleSpinBox_image_alpha.setSingleStep(0.1)
        self.doubleSpinBox_gradCAM_alpha.setDecimals(3)
        self.doubleSpinBox_gradCAM_alpha.setValue(0.0)

        self.doubleSpinBox_gradCAM_alpha.setObjectName("doubleSpinBox_gradCAM_alpha")
        self.gridLayout_3.addWidget(self.doubleSpinBox_gradCAM_alpha, 3, 1, 1, 1)
        self.gridLayout_4.addWidget(self.groupBox_gradCAM_Settings, 1, 0, 1, 2)
        self.pushButton_reset = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        self.pushButton_reset.setObjectName("pushButton_reset")
        self.gridLayout_4.addWidget(self.pushButton_reset, 2, 0, 1, 1)
        self.pushButton_update = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        self.pushButton_update.setObjectName("pushButton_update")
        self.gridLayout_4.addWidget(self.pushButton_update, 2, 1, 1, 1)
        self.scrollArea_settings.setWidget(self.scrollAreaWidgetContents)
        self.gridLayout_6.addWidget(self.scrollArea_settings, 2, 1, 1, 1)

        self.retranslateUi(Form_cm_interaction)
        QtCore.QMetaObject.connectSlotsByName(Form_cm_interaction)

    def retranslateUi(self, Form_cm_interaction):
        _translate = QtCore.QCoreApplication.translate
        Form_cm_interaction.setWindowTitle(_translate("Form_cm_interaction", "Show images/heatmaps"))
        self.groupBox_model.setTitle(_translate("Form_cm_interaction", "Model"))
        self.lineEdit_loadModel.setToolTip(_translate("Form_cm_interaction", "Enter path and filename of a history-file (.csv)"))
        self.pushButton_showSummary.setText(_translate("Form_cm_interaction", "Show summary"))
        self.label_inpImgSize.setText(_translate("Form_cm_interaction", "Input img. crop"))
        self.label_outpSize.setText(_translate("Form_cm_interaction", "Output Nr. of classes"))
        self.pushButton_toTensorB.setText(_translate("Form_cm_interaction", "To TensorBoard"))
        self.groupBox_imageShow.setTitle(_translate("Form_cm_interaction", "Image"))
        self.groupBox_image_Settings.setTitle(_translate("Form_cm_interaction", "Image"))
        self.label_image_alpha.setText(_translate("Form_cm_interaction", "Alpha"))
        self.groupBox_gradCAM_Settings.setTitle(_translate("Form_cm_interaction", "Grad-CAM"))
        self.label_gradCAM_targetClass.setText(_translate("Form_cm_interaction", "Class"))
        self.label_gradCAM_targetLayer.setText(_translate("Form_cm_interaction", "Layer"))
        self.label_gradCAM_colorMap.setText(_translate("Form_cm_interaction", "Colormap"))
        self.label_gradCAM_alpha.setText(_translate("Form_cm_interaction", "Alpha"))
        self.pushButton_reset.setText(_translate("Form_cm_interaction", "Reset"))
        self.pushButton_update.setText(_translate("Form_cm_interaction", "Update"))

        #Tooltips
        self.groupBox_model.setToolTip(_translate("Form", tooltips["groupBox_model"], None))
        self.lineEdit_loadModel.setToolTip(_translate("Form", tooltips["lineEdit_LoadModel_2"], None))
        self.pushButton_showSummary.setToolTip(_translate("Form", tooltips["pushButton_showSummary"], None))
        self.label_inpImgSize.setToolTip(_translate("Form", tooltips["label_inpImgSize"], None))
        self.spinBox_Crop_inpImgSize.setToolTip(_translate("Form", tooltips["label_inpImgSize"], None))
        self.label_outpSize.setToolTip(_translate("Form", tooltips["label_outpSize"], None))
        self.spinBox_outpSize.setToolTip(_translate("Form", tooltips["label_outpSize"], None))
        self.pushButton_toTensorB.setToolTip(_translate("Form", tooltips["pushButton_toTensorB"], None))
        self.groupBox_imageShow.setToolTip(_translate("Form", tooltips["groupBox_imageShow"], None))
        self.groupBox_image_Settings.setToolTip(_translate("Form", tooltips["groupBox_image_Settings"], None))
        self.label_image_alpha.setToolTip(_translate("Form", tooltips["label_image_alpha"], None))
        self.doubleSpinBox_image_alpha.setToolTip(_translate("Form", tooltips["label_image_alpha"], None))
        self.groupBox_gradCAM_Settings.setToolTip(_translate("Form", tooltips["groupBox_gradCAM_Settings"], None))
        self.label_gradCAM_targetClass.setToolTip(_translate("Form", tooltips["label_gradCAM_targetClass"], None))
        self.spinBox_gradCAM_targetClass.setToolTip(_translate("Form", tooltips["label_gradCAM_targetClass"], None))
        self.label_gradCAM_targetLayer.setToolTip(_translate("Form", tooltips["label_gradCAM_targetLayer"], None))
        self.comboBox_gradCAM_targetLayer.setToolTip(_translate("Form", tooltips["label_gradCAM_targetLayer"], None))
        self.label_gradCAM_colorMap.setToolTip(_translate("Form", tooltips["label_gradCAM_colorMap"], None))
        self.comboBox_gradCAM_colorMap.setToolTip(_translate("Form", tooltips["label_gradCAM_colorMap"], None))
        self.label_gradCAM_alpha.setToolTip(_translate("Form", tooltips["label_gradCAM_alpha"], None))
        self.doubleSpinBox_gradCAM_alpha.setToolTip(_translate("Form", tooltips["label_gradCAM_alpha"], None))
        self.pushButton_reset.setToolTip(_translate("Form", tooltips["pushButton_reset"], None))
        self.pushButton_update.setToolTip(_translate("Form", tooltips["pushButton_update"], None))



    def gradCAM_on_off(self,on_or_off):
        if on_or_off==False:#it is switched off
            #set image_alpha to 1
            self.doubleSpinBox_image_alpha.setValue(1)
        if on_or_off==True:#it is switched on
            #set image_alpha and gradCAM_alpha to 0.5
            self.doubleSpinBox_image_alpha.setValue(0.5)
            self.doubleSpinBox_gradCAM_alpha.setValue(0.5)

    def image_on_off(self,on_or_off):
        if on_or_off==False:#it is switched off
            self.doubleSpinBox_image_alpha.setValue(0)


class popup_cm_modelsummary(QtWidgets.QWidget):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(300, 300)
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        self.textBrowser_modelsummary = QtWidgets.QTextBrowser(Form)
        self.textBrowser_modelsummary.setObjectName("textBrowser_modelsummary")
        self.gridLayout.addWidget(self.textBrowser_modelsummary, 0, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Model summary"))



class popup_lrfinder(QtWidgets.QWidget):
    def setupUi(self, Form_LrFinder):
        Form_LrFinder.setObjectName("Form_LrFinder")
        Form_LrFinder.resize(740, 740)
        self.gridLayout_3 = QtWidgets.QGridLayout(Form_LrFinder)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.scrollArea_LrFinder = QtWidgets.QScrollArea(Form_LrFinder)
        self.scrollArea_LrFinder.setWidgetResizable(True)
        self.scrollArea_LrFinder.setObjectName("scrollArea_LrFinder")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 740, 740))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.gridLayout_9 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents)
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.splitter = QtWidgets.QSplitter(self.scrollAreaWidgetContents)
        self.splitter.setOrientation(QtCore.Qt.Vertical)
        self.splitter.setObjectName("splitter")
        self.groupBox_model = QtWidgets.QGroupBox(self.splitter)
        self.groupBox_model.setMaximumSize(QtCore.QSize(16777215, 150))
        self.groupBox_model.setObjectName("groupBox_model")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox_model)
        self.gridLayout.setObjectName("gridLayout")
        
        
        self.label_inpImgSize = QtWidgets.QLabel(self.groupBox_model)
        self.label_inpImgSize.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_inpImgSize.setObjectName("label_inpImgSize")
        self.gridLayout.addWidget(self.label_inpImgSize, 1, 0, 1, 1)
        self.label_expt_loss = QtWidgets.QLabel(self.groupBox_model)
        self.label_expt_loss.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_expt_loss.setObjectName("label_expt_loss")
        self.gridLayout.addWidget(self.label_expt_loss, 2, 0, 1, 1)
        self.spinBox_Crop_inpImgSize = QtWidgets.QSpinBox(self.groupBox_model)
        self.spinBox_Crop_inpImgSize.setEnabled(False)
        self.spinBox_Crop_inpImgSize.setObjectName("spinBox_Crop_inpImgSize")
        self.gridLayout.addWidget(self.spinBox_Crop_inpImgSize, 1, 1, 1, 1)
        self.comboBox_expt_loss = QtWidgets.QComboBox(self.groupBox_model)
        self.comboBox_expt_loss.setEnabled(False)
        self.comboBox_expt_loss.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.comboBox_expt_loss.setObjectName("comboBox_expt_loss")
        self.comboBox_expt_loss.addItem("")
        self.comboBox_expt_loss.addItem("")
        self.comboBox_expt_loss.addItem("")
        self.comboBox_expt_loss.addItem("")
        self.comboBox_expt_loss.addItem("")
        self.comboBox_expt_loss.addItem("")
        self.comboBox_expt_loss.addItem("")
        self.comboBox_expt_loss.addItem("")
        self.comboBox_expt_loss.addItem("")
        self.comboBox_expt_loss.addItem("")
        self.comboBox_expt_loss.addItem("")
        self.comboBox_expt_loss.addItem("")
        self.comboBox_expt_loss.addItem("")
        self.comboBox_expt_loss.addItem("")
        self.comboBox_expt_loss.addItem("")
        self.comboBox_expt_loss.addItem("")
        self.gridLayout.addWidget(self.comboBox_expt_loss, 2, 1, 1, 1)
        self.label_colorMode = QtWidgets.QLabel(self.groupBox_model)
        self.label_colorMode.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_colorMode.setObjectName("label_colorMode")
        self.gridLayout.addWidget(self.label_colorMode, 1, 2, 1, 1)
        self.label_optimizer = QtWidgets.QLabel(self.groupBox_model)
        self.label_optimizer.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_optimizer.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_optimizer.setObjectName("label_optimizer")
        self.gridLayout.addWidget(self.label_optimizer, 2, 2, 1, 1)
        self.comboBox_colorMode = QtWidgets.QComboBox(self.groupBox_model)
        self.comboBox_colorMode.setEnabled(False)
        self.comboBox_colorMode.setObjectName("comboBox_colorMode")
        self.gridLayout.addWidget(self.comboBox_colorMode, 1, 3, 1, 1)
        self.comboBox_optimizer = QtWidgets.QComboBox(self.groupBox_model)
        self.comboBox_optimizer.setEnabled(False)
        self.comboBox_optimizer.setObjectName("comboBox_optimizer")
        self.comboBox_optimizer.addItem("")
        self.comboBox_optimizer.addItem("")
        self.comboBox_optimizer.addItem("")
        self.comboBox_optimizer.addItem("")
        self.comboBox_optimizer.addItem("")
        self.comboBox_optimizer.addItem("")
        self.comboBox_optimizer.addItem("")
        self.gridLayout.addWidget(self.comboBox_optimizer, 2, 3, 1, 1)
        self.lineEdit_loadModel = QtWidgets.QLineEdit(self.groupBox_model)
        self.lineEdit_loadModel.setEnabled(False)
        self.lineEdit_loadModel.setObjectName("lineEdit_loadModel")
        self.gridLayout.addWidget(self.lineEdit_loadModel, 0, 0, 1, 4)
        self.widget = QtWidgets.QWidget(self.splitter)
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.groupBox_LrSettings = QtWidgets.QGroupBox(self.widget)
        self.groupBox_LrSettings.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.groupBox_LrSettings.setCheckable(False)
        self.groupBox_LrSettings.setObjectName("groupBox_LrSettings")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.groupBox_LrSettings)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.label_startLR = QtWidgets.QLabel(self.groupBox_LrSettings)
        self.label_startLR.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_startLR.setObjectName("label_startLR")
        self.gridLayout_5.addWidget(self.label_startLR, 0, 0, 1, 1)
        self.lineEdit_startLr = QtWidgets.QLineEdit(self.groupBox_LrSettings)
        self.lineEdit_startLr.setObjectName("lineEdit_startLr")
        
        validator = QtGui.QRegExpValidator(QtCore.QRegExp("^[0-9 . , e -]+$")) #validator allows numbers, dots, commas, e and -
        self.lineEdit_startLr.setValidator(validator)                     
        
        self.gridLayout_5.addWidget(self.lineEdit_startLr, 0, 1, 1, 3)
        self.label_percDataT = QtWidgets.QLabel(self.groupBox_LrSettings)
        self.label_percDataT.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_percDataT.setObjectName("label_percDataT")
        self.gridLayout_5.addWidget(self.label_percDataT, 0, 4, 1, 1)
        self.doubleSpinBox_percDataT = QtWidgets.QDoubleSpinBox(self.groupBox_LrSettings)
        self.doubleSpinBox_percDataT.setMaximum(10000.0)
        self.doubleSpinBox_percDataT.setProperty("value", 100.0)
        self.doubleSpinBox_percDataT.setObjectName("doubleSpinBox_percDataT")
        self.gridLayout_5.addWidget(self.doubleSpinBox_percDataT, 0, 5, 1, 1)
        self.label_stopLr = QtWidgets.QLabel(self.groupBox_LrSettings)
        self.label_stopLr.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_stopLr.setObjectName("label_stopLr")
        self.gridLayout_5.addWidget(self.label_stopLr, 1, 0, 1, 1)
        self.lineEdit_stopLr = QtWidgets.QLineEdit(self.groupBox_LrSettings)
        self.lineEdit_stopLr.setClearButtonEnabled(False)
        self.lineEdit_stopLr.setObjectName("lineEdit_stopLr")
        validator = QtGui.QRegExpValidator(QtCore.QRegExp("^[0-9 . , e -]+$")) #validator allows numbers, dots, commas, e and -
        self.lineEdit_stopLr.setValidator(validator)

        self.gridLayout_5.addWidget(self.lineEdit_stopLr, 1, 1, 1, 3)
        self.label_percDataV = QtWidgets.QLabel(self.groupBox_LrSettings)
        self.label_percDataV.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_percDataV.setObjectName("label_percDataV")
        self.gridLayout_5.addWidget(self.label_percDataV, 1, 4, 1, 1)
        self.doubleSpinBox_percDataV = QtWidgets.QDoubleSpinBox(self.groupBox_LrSettings)
        self.doubleSpinBox_percDataV.setMaximum(10000.0)
        self.doubleSpinBox_percDataV.setProperty("value", 25.0)
        self.doubleSpinBox_percDataV.setEnabled(False)
        
        self.doubleSpinBox_percDataV.setObjectName("doubleSpinBox_percDataV")
        self.gridLayout_5.addWidget(self.doubleSpinBox_percDataV, 1, 5, 1, 1)
        self.label_batchSize = QtWidgets.QLabel(self.groupBox_LrSettings)
        self.label_batchSize.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_batchSize.setObjectName("label_batchSize")
        self.gridLayout_5.addWidget(self.label_batchSize, 2, 0, 1, 1)
        self.spinBox_batchSize = QtWidgets.QSpinBox(self.groupBox_LrSettings)
        self.spinBox_batchSize.setMaximum(999999999)
        self.spinBox_batchSize.setProperty("value", 32)
        self.spinBox_batchSize.setObjectName("spinBox_batchSize")
        self.gridLayout_5.addWidget(self.spinBox_batchSize, 2, 1, 1, 1)
        self.label_epochs = QtWidgets.QLabel(self.groupBox_LrSettings)
        self.label_epochs.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_epochs.setObjectName("label_epochs")
        self.gridLayout_5.addWidget(self.label_epochs, 2, 2, 1, 1)
        self.spinBox_epochs = QtWidgets.QSpinBox(self.groupBox_LrSettings)
        self.spinBox_epochs.setMinimum(1)
        self.spinBox_epochs.setMaximum(1000)
        self.spinBox_epochs.setProperty("value", 5)
        self.spinBox_epochs.setObjectName("spinBox_epochs")
        self.gridLayout_5.addWidget(self.spinBox_epochs, 2, 3, 1, 1)
        self.label_stepsPerEpoch = QtWidgets.QLabel(self.groupBox_LrSettings)
        self.label_stepsPerEpoch.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_stepsPerEpoch.setObjectName("label_stepsPerEpoch")
        self.gridLayout_5.addWidget(self.label_stepsPerEpoch, 2, 4, 1, 1)
        self.spinBox_stepsPerEpoch = QtWidgets.QSpinBox(self.groupBox_LrSettings)
        self.spinBox_stepsPerEpoch.setEnabled(False)
        self.spinBox_stepsPerEpoch.setMaximum(999999999)
        self.spinBox_stepsPerEpoch.setProperty("value", 4)
        self.spinBox_stepsPerEpoch.setObjectName("spinBox_stepsPerEpoch")
        self.gridLayout_5.addWidget(self.spinBox_stepsPerEpoch, 2, 5, 1, 1)
        self.pushButton_LrReset = QtWidgets.QPushButton(self.groupBox_LrSettings)
        self.pushButton_LrReset.setMaximumSize(QtCore.QSize(100, 16777215))

        self.pushButton_LrReset.setObjectName("pushButton_LrReset")
        self.gridLayout_5.addWidget(self.pushButton_LrReset, 3, 0, 1, 2)
        self.pushButton_LrFindRun = QtWidgets.QPushButton(self.groupBox_LrSettings)
        font = QtGui.QFont()
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(True)
        self.pushButton_LrFindRun.setFont(font)
        self.pushButton_LrFindRun.setObjectName("pushButton_LrFindRun")
        self.gridLayout_5.addWidget(self.pushButton_LrFindRun, 3, 5, 1, 1)
        self.horizontalLayout.addWidget(self.groupBox_LrSettings)
        self.groupBox_design = QtWidgets.QGroupBox(self.widget)
        self.groupBox_design.setObjectName("groupBox_design")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox_design)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.checkBox_valMetrics = QtWidgets.QCheckBox(self.groupBox_design)
        self.checkBox_valMetrics.setObjectName("checkBox_valMetrics")
        self.checkBox_valMetrics.toggled.connect(self.valMetrics)

        self.gridLayout_2.addWidget(self.checkBox_valMetrics, 1, 0, 1, 2)
        
        self.pushButton_color = QtWidgets.QPushButton(self.groupBox_design)
        self.pushButton_color.setStyleSheet("background-color: blue"+";")
        self.pushButton_color.setObjectName("pushButton_color")
        self.pushButton_color.setObjectName("pushButton_color")
        self.gridLayout_2.addWidget(self.pushButton_color, 2, 0, 1, 2)
        self.pushButton_color.clicked.connect(self.lr_color_picker)





        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_lineWidth = QtWidgets.QLabel(self.groupBox_design)
        self.label_lineWidth.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_lineWidth.setObjectName("label_lineWidth")
        self.horizontalLayout_2.addWidget(self.label_lineWidth)
        self.spinBox_lineWidth = QtWidgets.QSpinBox(self.groupBox_design)
        self.spinBox_lineWidth.setMinimum(1)
        self.spinBox_lineWidth.setMaximum(100)
        self.spinBox_lineWidth.setProperty("value", 6)
        self.spinBox_lineWidth.setObjectName("spinBox_lineWidth")
        self.horizontalLayout_2.addWidget(self.spinBox_lineWidth)
        self.checkBox_smooth = QtWidgets.QCheckBox(self.groupBox_design)
        self.checkBox_smooth.setObjectName("checkBox_smooth")
        self.checkBox_smooth.setChecked(True)
        self.horizontalLayout_2.addWidget(self.checkBox_smooth)
        self.gridLayout_2.addLayout(self.horizontalLayout_2, 3, 0, 1, 1)
        
        self.comboBox_metric = QtWidgets.QComboBox(self.groupBox_design)
        self.comboBox_metric.setObjectName("comboBox_metric")
        
        self.gridLayout_2.addWidget(self.comboBox_metric, 0, 0, 1, 2)
        self.horizontalLayout.addWidget(self.groupBox_design)
        self.groupBox_LrPlotting = QtWidgets.QGroupBox(self.splitter)
        self.groupBox_LrPlotting.setMinimumSize(QtCore.QSize(0, 250))
        self.groupBox_LrPlotting.setObjectName("groupBox_LrPlotting")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.groupBox_LrPlotting)
        self.gridLayout_7.setObjectName("gridLayout_7")
        
        self.widget_LrPlotting = pg.GraphicsLayoutWidget(self.groupBox_LrPlotting)
        self.widget_LrPlotting.setMinimumSize(QtCore.QSize(0, 0))
        self.widget_LrPlotting.setObjectName("widget_LrPlotting")

        self.lr_plot = self.widget_LrPlotting.addPlot()
        self.lr_plot.showGrid(x=True,y=True)
        self.lr_plot.setLabel('bottom', 'Learning rate (log scale)', units='')
        self.lr_plot.setLogMode(x=True, y=False)
        self.lr_plot.addLegend()

        self.gridLayout_7.addWidget(self.widget_LrPlotting, 0, 0, 1, 1)
        self.layoutWidget = QtWidgets.QWidget(self.splitter)
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout_8.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.groupBox_singleLr = QtWidgets.QGroupBox(self.layoutWidget)
        self.groupBox_singleLr.setCheckable(True)
        self.groupBox_singleLr.setChecked(False)
        self.groupBox_singleLr.setEnabled(False)#intially, the groupbox is disabled. Will be enabled after the lr_find algorithm did run
        self.groupBox_singleLr.setObjectName("groupBox_singleLr")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.groupBox_singleLr)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.lineEdit_singleLr = QtWidgets.QLineEdit(self.groupBox_singleLr)
        self.lineEdit_singleLr.setReadOnly(False)
        self.lineEdit_singleLr.setClearButtonEnabled(True)
        self.lineEdit_singleLr.setObjectName("lineEdit_singleLr")
        self.gridLayout_6.addWidget(self.lineEdit_singleLr, 0, 0, 1, 2)
        self.pushButton_singleReset = QtWidgets.QPushButton(self.groupBox_singleLr)

        self.pushButton_singleReset.setObjectName("pushButton_singleReset")
        self.gridLayout_6.addWidget(self.pushButton_singleReset, 1, 0, 1, 1)
        self.pushButton_singleAccept = QtWidgets.QPushButton(self.groupBox_singleLr)

        self.pushButton_singleAccept.setObjectName("pushButton_singleAccept")
        self.gridLayout_6.addWidget(self.pushButton_singleAccept, 1, 1, 1, 1)
        self.gridLayout_8.addWidget(self.groupBox_singleLr, 0, 0, 1, 1)
        self.groupBox_LrRange = QtWidgets.QGroupBox(self.layoutWidget)
        self.groupBox_LrRange.setCheckable(True)
        self.groupBox_LrRange.setChecked(False)
        self.groupBox_LrRange.setEnabled(False)#intially, the groupbox is disabled. Will be enabled after the lr_find algorithm did run
        self.groupBox_LrRange.setObjectName("groupBox_LrRange")
        
        self.gridLayout_4 = QtWidgets.QGridLayout(self.groupBox_LrRange)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_LrMin = QtWidgets.QLabel(self.groupBox_LrRange)
        self.label_LrMin.setObjectName("label_LrMin")
        self.gridLayout_4.addWidget(self.label_LrMin, 0, 0, 1, 1)
        self.lineEdit_LrMin = QtWidgets.QLineEdit(self.groupBox_LrRange)
        self.lineEdit_LrMin.setReadOnly(False)
        self.lineEdit_LrMin.setClearButtonEnabled(True)
        self.lineEdit_LrMin.setObjectName("lineEdit_LrMin")
        self.gridLayout_4.addWidget(self.lineEdit_LrMin, 0, 1, 1, 1)
        self.label_LrMax = QtWidgets.QLabel(self.groupBox_LrRange)
        self.label_LrMax.setObjectName("label_LrMax")
        self.gridLayout_4.addWidget(self.label_LrMax, 0, 2, 1, 1)
        self.lineEdit_LrMax = QtWidgets.QLineEdit(self.groupBox_LrRange)
        self.lineEdit_LrMax.setClearButtonEnabled(True)
        self.lineEdit_LrMax.setObjectName("lineEdit_LrMax")
        self.gridLayout_4.addWidget(self.lineEdit_LrMax, 0, 3, 1, 1)
        self.pushButton_rangeAccept = QtWidgets.QPushButton(self.groupBox_LrRange)

        self.pushButton_rangeAccept.setObjectName("pushButton_rangeAccept")
        self.gridLayout_4.addWidget(self.pushButton_rangeAccept, 1, 3, 1, 1)
        self.pushButton_rangeReset = QtWidgets.QPushButton(self.groupBox_LrRange)

        self.pushButton_rangeReset.setObjectName("pushButton_rangeReset")
        self.gridLayout_4.addWidget(self.pushButton_rangeReset, 1, 0, 1, 2)
        self.gridLayout_8.addWidget(self.groupBox_LrRange, 0, 1, 1, 1)
        self.gridLayout_9.addWidget(self.splitter, 0, 0, 1, 1)
        self.scrollArea_LrFinder.setWidget(self.scrollAreaWidgetContents)
        self.gridLayout_3.addWidget(self.scrollArea_LrFinder, 0, 1, 1, 1)


        ######Icons########
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(os.path.join(dir_root,"art",Default_dict["Icon theme"],"reset.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_LrReset.setIcon(icon)
        self.pushButton_singleReset.setIcon(icon)
        self.pushButton_rangeReset.setIcon(icon)

        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(os.path.join(dir_root,"art",Default_dict["Icon theme"],"thumb.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_LrFindRun.setIcon(icon1)
        self.pushButton_rangeAccept.setIcon(icon1)
        self.pushButton_singleAccept.setIcon(icon1)
        
        
        #######Connections#############
        self.checkBox_valMetrics.clicked['bool'].connect(self.doubleSpinBox_percDataV.setEnabled)



        self.retranslateUi(Form_LrFinder)
        QtCore.QMetaObject.connectSlotsByName(Form_LrFinder)

    def retranslateUi(self, Form_LrFinder):
        _translate = QtCore.QCoreApplication.translate
        Form_LrFinder.setWindowTitle(_translate("Form_LrFinder", "LR Screening", None))
        self.groupBox_model.setTitle(_translate("Form_LrFinder", "Model", None))
        self.label_colorMode.setText(_translate("Form_LrFinder", "Color Mode"))              
        self.label_inpImgSize.setText(_translate("Form_LrFinder", "Input img. crop", None))
        self.lineEdit_loadModel.setToolTip(_translate("Form_LrFinder", tooltips["lineEdit_LoadModel_3"], None))
        self.comboBox_optimizer.setItemText(0, _translate("Form_LrFinder", "Adam", None))
        self.comboBox_optimizer.setItemText(1, _translate("Form_LrFinder", "SGD", None))
        self.comboBox_optimizer.setItemText(2, _translate("Form_LrFinder", "RMSprop", None))
        self.comboBox_optimizer.setItemText(3, _translate("Form_LrFinder", "Adagrad", None))
        self.comboBox_optimizer.setItemText(4, _translate("Form_LrFinder", "Adadelta", None))
        self.comboBox_optimizer.setItemText(5, _translate("Form_LrFinder", "Adamax", None))
        self.comboBox_optimizer.setItemText(6, _translate("Form_LrFinder", "Nadam", None))
        self.label_expt_loss.setText(_translate("Form_LrFinder", "Loss", None))
        self.label_optimizer.setText(_translate("Form_LrFinder", "Optimizer", None))
        self.comboBox_expt_loss.setItemText(0, _translate("Form_LrFinder", "categorical_crossentropy", None))
        self.comboBox_expt_loss.setItemText(1, _translate("Form_LrFinder", "sparse_categorical_crossentropy", None))
        self.comboBox_expt_loss.setItemText(2, _translate("Form_LrFinder", "mean_squared_error", None))
        self.comboBox_expt_loss.setItemText(3, _translate("Form_LrFinder", "mean_absolute_error", None))
        self.comboBox_expt_loss.setItemText(4, _translate("Form_LrFinder", "mean_absolute_percentage_error", None))
        self.comboBox_expt_loss.setItemText(5, _translate("Form_LrFinder", "mean_squared_logarithmic_error", None))
        self.comboBox_expt_loss.setItemText(6, _translate("Form_LrFinder", "squared_hinge", None))
        self.comboBox_expt_loss.setItemText(7, _translate("Form_LrFinder", "hinge", None))
        self.comboBox_expt_loss.setItemText(8, _translate("Form_LrFinder", "categorical_hinge", None))
        self.comboBox_expt_loss.setItemText(9, _translate("Form_LrFinder", "logcosh", None))
        self.comboBox_expt_loss.setItemText(10, _translate("Form_LrFinder", "huber_loss", None))
        self.comboBox_expt_loss.setItemText(11, _translate("Form_LrFinder", "binary_crossentropy", None))
        self.comboBox_expt_loss.setItemText(12, _translate("Form_LrFinder", "kullback_leibler_divergence", None))
        self.comboBox_expt_loss.setItemText(13, _translate("Form_LrFinder", "poisson", None))
        self.comboBox_expt_loss.setItemText(14, _translate("Form_LrFinder", "cosine_proximity", None))
        self.comboBox_expt_loss.setItemText(15, _translate("Form_LrFinder", "is_categorical_crossentropy", None))
        self.groupBox_LrSettings.setTitle(_translate("Form_LrFinder", "LR screening settings", None))
        self.label_startLR.setText(_translate("Form_LrFinder", "Start LR", None))
        self.lineEdit_startLr.setText(_translate("Form_LrFinder", "1e-10", None))
        self.label_percDataT.setText(_translate("Form_LrFinder", "% of data (T)", None))
        self.label_percDataV.setText(_translate("Form_LrFinder", "% of data (V)", None))

        self.label_batchSize.setText(_translate("Form_LrFinder", "Batch size", None))
        self.label_epochs.setText(_translate("Form_LrFinder", "Epochs", None))
        self.groupBox_design.setTitle(_translate("Form_LrFinder", "Plotting options", None))
        self.checkBox_valMetrics.setText(_translate("Form_LrFinder", "Validation metrics"))
        self.checkBox_valMetrics.setToolTip(_translate("MainWindow", tooltips["checkBox_valMetrics"],None))

        self.pushButton_color.setText(_translate("Form_LrFinder", "Color", None))
        self.label_lineWidth.setText(_translate("Form_LrFinder", "Line width", None))
        self.checkBox_smooth.setText(_translate("Form_LrFinder", "Smooth", None))
        self.checkBox_smooth.setToolTip(_translate("MainWindow", tooltips["checkBox_smooth"],None))

        self.comboBox_metric.addItem("Loss")
        self.comboBox_metric.addItem("Loss 1st derivative")
        self.comboBox_metric.addItem("Accuracy")
        self.comboBox_metric.addItem("Accuracy 1st derivative")

        self.label_stepsPerEpoch.setText(_translate("Form_LrFinder", "Steps/Epoch", None))
        self.pushButton_LrReset.setText(_translate("Form_LrFinder", "Reset", None))
        self.pushButton_LrFindRun.setText(_translate("Form_LrFinder", "Run", None))
        self.label_stopLr.setText(_translate("Form_LrFinder", "Stop LR", None))
        self.lineEdit_stopLr.setText(_translate("Form_LrFinder", "0.1", None))
        self.groupBox_LrPlotting.setTitle(_translate("Form_LrFinder", "LR screening", None))
        self.groupBox_singleLr.setTitle(_translate("Form_LrFinder", "Single LR", None))
        self.pushButton_singleReset.setText(_translate("Form_LrFinder", "Reset", None))
        self.pushButton_singleAccept.setText(_translate("Form_LrFinder", "Accept", None))
        self.groupBox_LrRange.setTitle(_translate("Form_LrFinder", "LR range", None))
        self.label_LrMin.setText(_translate("Form_LrFinder", "Min", None))
        self.label_LrMax.setText(_translate("Form_LrFinder", "Max", None))
        self.pushButton_rangeAccept.setText(_translate("Form_LrFinder", "Accept", None))
        self.pushButton_rangeReset.setText(_translate("Form_LrFinder", "Reset", None))

        ###############Tooltips##################
        self.groupBox_model.setToolTip(_translate("MainWindow", tooltips["groupBox_model"],None))
        self.label_inpImgSize.setToolTip(_translate("MainWindow", tooltips["label_inpImgSize"],None))
        self.spinBox_Crop_inpImgSize.setToolTip(_translate("MainWindow", tooltips["label_inpImgSize"],None))
        self.label_colorMode.setToolTip(_translate("MainWindow", tooltips["label_colorMode"],None))
        self.comboBox_colorMode.setToolTip(_translate("MainWindow", tooltips["label_colorMode"],None))
        self.label_expt_loss.setToolTip(_translate("MainWindow", tooltips["label_expt_loss"],None))
        self.label_expt_loss.setToolTip(_translate("MainWindow", tooltips["label_expt_loss"],None))
        self.label_optimizer.setToolTip(_translate("MainWindow", tooltips["label_optimizer"],None))
        self.comboBox_optimizer.setToolTip(_translate("MainWindow", tooltips["label_optimizer"],None))

        self.label_startLR.setToolTip(_translate("MainWindow", tooltips["label_startLR"],None))
        self.lineEdit_startLr.setToolTip(_translate("MainWindow", tooltips["label_startLR"],None))
        self.label_stopLr.setToolTip(_translate("MainWindow", tooltips["label_stopLr"],None))
        self.lineEdit_stopLr.setToolTip(_translate("MainWindow", tooltips["label_stopLr"],None))

        self.label_percDataT.setToolTip(_translate("MainWindow", tooltips["label_percData"],None))
        self.doubleSpinBox_percDataT.setToolTip(_translate("MainWindow", tooltips["label_percData"],None))
        self.label_percDataV.setToolTip(_translate("MainWindow", tooltips["label_percData"],None))
        self.doubleSpinBox_percDataV.setToolTip(_translate("MainWindow", tooltips["label_percData"],None))

        self.label_batchSize.setToolTip(_translate("MainWindow", tooltips["label_batchSize"],None))
        self.spinBox_batchSize.setToolTip(_translate("MainWindow", tooltips["label_batchSize"],None))
        
        self.label_stepsPerEpoch.setToolTip(_translate("MainWindow", tooltips["label_stepsPerEpoch"],None))
        self.spinBox_stepsPerEpoch.setToolTip(_translate("MainWindow", tooltips["label_stepsPerEpoch"],None))


        self.pushButton_LrReset.setToolTip(_translate("MainWindow", tooltips["pushButton_LrReset"],None))
        self.pushButton_color.setToolTip(_translate("MainWindow", tooltips["pushButton_color"],None))
        self.label_lineWidth.setToolTip(_translate("MainWindow", tooltips["label_lineWidth"],None))
        self.spinBox_lineWidth.setToolTip(_translate("MainWindow", tooltips["label_lineWidth"],None))

        self.label_epochs.setToolTip(_translate("MainWindow", tooltips["label_epochs_lrfind"],None))
        self.spinBox_epochs.setToolTip(_translate("MainWindow", tooltips["label_epochs_lrfind"],None))
        self.pushButton_LrFindRun.setToolTip(_translate("MainWindow", tooltips["pushButton_LrFindRun"],None))
        self.groupBox_LrSettings.setToolTip(_translate("MainWindow", tooltips["groupBox_LrSettings"],None))
        self.widget_LrPlotting.setToolTip(_translate("MainWindow", tooltips["groupBox_LrSettings"],None))

        self.groupBox_singleLr.setToolTip(_translate("MainWindow", tooltips["groupBox_singleLr"],None))
        self.groupBox_LrRange.setToolTip(_translate("MainWindow", tooltips["groupBox_LrRange"],None))


    

    def lr_color_picker(self):
        color = QtGui.QColorDialog.getColor()
        if color.getRgb()==(0, 0, 0, 255):#no black!
            return
        else:
            #self.pushButton_color.setBackground(color)
            self.pushButton_color.setStyleSheet("background-color: "+ color.name()+";")
    
    def valMetrics(self):
        valMetrics_onOff = self.checkBox_valMetrics.isChecked()
        self.comboBox_metric.clear()
        
        self.comboBox_metric.addItem("Loss")
        self.comboBox_metric.addItem("Loss 1st derivative")
        self.comboBox_metric.addItem("Accuracy")
        self.comboBox_metric.addItem("Accuracy 1st derivative")
        
        if valMetrics_onOff==True:    
            self.comboBox_metric.addItem("Val. loss")
            self.comboBox_metric.addItem("Val. loss 1st derivative")
            self.comboBox_metric.addItem("Val. accuracy")
            self.comboBox_metric.addItem("Val. accuracy 1st derivative")


class popup_lrplot(QtWidgets.QWidget):
    def setupUi(self, LearningRatePlot):
        LearningRatePlot.setObjectName("LearningRatePlot")
        LearningRatePlot.resize(534, 451)
        self.gridLayout = QtWidgets.QGridLayout(LearningRatePlot)
        self.gridLayout.setObjectName("gridLayout")
        self.splitter = QtWidgets.QSplitter(LearningRatePlot)
        self.splitter.setOrientation(QtCore.Qt.Vertical)
        self.splitter.setObjectName("splitter")
        self.groupBox_design = QtWidgets.QGroupBox(self.splitter)
        self.groupBox_design.setObjectName("groupBox_design")
        self.gridLayout_10 = QtWidgets.QGridLayout(self.groupBox_design)
        self.gridLayout_10.setContentsMargins(-1, 2, -1, 2)
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.label_lineWidth = QtWidgets.QLabel(self.groupBox_design)
        self.label_lineWidth.setObjectName("label_lineWidth")
        self.gridLayout_10.addWidget(self.label_lineWidth, 0, 1, 1, 1)
        self.pushButton_color = QtWidgets.QPushButton(self.groupBox_design)
        self.pushButton_color.setObjectName("pushButton_color")
        self.gridLayout_10.addWidget(self.pushButton_color, 0, 0, 1, 1)
        self.spinBox_lineWidth = QtWidgets.QSpinBox(self.groupBox_design)
        self.spinBox_lineWidth.setMinimum(1)
        self.spinBox_lineWidth.setMaximum(100)
        self.spinBox_lineWidth.setProperty("value", 6)
        self.spinBox_lineWidth.setObjectName("spinBox_lineWidth")
        self.gridLayout_10.addWidget(self.spinBox_lineWidth, 0, 2, 1, 1)
        self.pushButton_refreshPlot = QtWidgets.QPushButton(self.groupBox_design)
        self.pushButton_refreshPlot.setObjectName("pushButton_refreshPlot")
        self.gridLayout_10.addWidget(self.pushButton_refreshPlot, 0, 3, 1, 1)
        self.spinBox_totalEpochs = QtWidgets.QSpinBox(self.groupBox_design)
        self.spinBox_totalEpochs.setEnabled(False)
        self.spinBox_totalEpochs.setMaximum(999999999)
        self.spinBox_totalEpochs.setObjectName("spinBox_totalEpochs")
        self.gridLayout_10.addWidget(self.spinBox_totalEpochs, 0, 5, 1, 1)
        self.label_totalEpochs = QtWidgets.QLabel(self.groupBox_design)
        self.label_totalEpochs.setObjectName("label_totalEpochs")
        self.gridLayout_10.addWidget(self.label_totalEpochs, 0, 4, 1, 1)
        self.groupBox = QtWidgets.QGroupBox(self.splitter)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.textBrowser_lrSettings = QtWidgets.QTextBrowser(self.groupBox)
        self.textBrowser_lrSettings.setObjectName("textBrowser_lrSettings")
        self.gridLayout_2.addWidget(self.textBrowser_lrSettings, 0, 0, 1, 1)
        self.groupBox_plottingRegion = QtWidgets.QGroupBox(self.splitter)
        self.groupBox_plottingRegion.setObjectName("groupBox_plottingRegion")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox_plottingRegion)
        self.gridLayout_3.setObjectName("gridLayout_3")


        self.widget_LrPlotting = pg.GraphicsLayoutWidget(self.groupBox_plottingRegion)
        self.widget_LrPlotting.setMinimumSize(QtCore.QSize(200, 200))
        self.widget_LrPlotting.setObjectName("widget_LrPlotting")
        self.gridLayout_3.addWidget(self.widget_LrPlotting, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.splitter, 0, 0, 1, 1)



        ####Manual additions####
        self.pushButton_color.setStyleSheet("background-color: blue"+";")
        self.pushButton_color.clicked.connect(self.lr_color_picker)     
        self.lr_plot = self.widget_LrPlotting.addPlot()
        self.lr_plot.showGrid(x=True,y=True)
        self.lr_plot.setLabel('bottom', 'Epoch', units='')
        self.lr_plot.setLabel('left', 'Learning rate', units='')
        self.lr_plot.setLogMode(x=False, y=False)
        
        self.retranslateUi(LearningRatePlot)
        QtCore.QMetaObject.connectSlotsByName(LearningRatePlot)

    def retranslateUi(self, LearningRatePlot):
        _translate = QtCore.QCoreApplication.translate
        LearningRatePlot.setWindowTitle(_translate("LearningRatePlot", "Learning rate plotting",None))
        self.groupBox_design.setTitle(_translate("LearningRatePlot", "Plot design",None))
        self.label_lineWidth.setText(_translate("LearningRatePlot", "Line width",None))
        self.pushButton_color.setText(_translate("LearningRatePlot", "Color",None))
        self.pushButton_refreshPlot.setText(_translate("LearningRatePlot", "Refresh",None))
        self.label_totalEpochs.setText(_translate("LearningRatePlot", "Total epochs",None))
        self.groupBox.setTitle(_translate("LearningRatePlot", "LR settings",None))
        self.groupBox_plottingRegion.setTitle(_translate("LearningRatePlot", "Plotting region",None))

    def lr_color_picker(self):
        color = QtGui.QColorDialog.getColor()
        if color.getRgb()==(0, 0, 0, 255):#no black!
            return
        else:
            #self.pushButton_color.setBackground(color)
            self.pushButton_color.setStyleSheet("background-color: "+ color.name()+";")


class Ui_Clr_settings(QtWidgets.QWidget):
    def setupUi(self, Clr_settings):
        Clr_settings.setObjectName("Clr_settings")
        Clr_settings.resize(215, 108)
        self.gridLayout = QtWidgets.QGridLayout(Clr_settings)
        self.gridLayout.setObjectName("gridLayout")
        self.label_stepSize = QtWidgets.QLabel(Clr_settings)
        self.label_stepSize.setObjectName("label_stepSize")
        self.gridLayout.addWidget(self.label_stepSize, 0, 0, 1, 1)
        self.spinBox_stepSize = QtWidgets.QSpinBox(Clr_settings)
        self.spinBox_stepSize.setMinimum(1)
        self.spinBox_stepSize.setMaximum(9999999)
        self.spinBox_stepSize.setObjectName("spinBox_stepSize")
        self.gridLayout.addWidget(self.spinBox_stepSize, 0, 1, 1, 2)
        self.label_gamma = QtWidgets.QLabel(Clr_settings)
        self.label_gamma.setObjectName("label_gamma")
        self.gridLayout.addWidget(self.label_gamma, 1, 0, 1, 1)
        self.doubleSpinBox_gamma = QtWidgets.QDoubleSpinBox(Clr_settings)
        self.doubleSpinBox_gamma.setDecimals(9)
        self.doubleSpinBox_gamma.setSingleStep(0.001)
        self.doubleSpinBox_gamma.setObjectName("doubleSpinBox_gamma")
        self.gridLayout.addWidget(self.doubleSpinBox_gamma, 1, 1, 1, 2)
        self.pushButton_cancel = QtWidgets.QPushButton(Clr_settings)
        self.pushButton_cancel.setObjectName("pushButton_cancel")
        self.gridLayout.addWidget(self.pushButton_cancel, 2, 0, 1, 2)
        self.pushButton_ok = QtWidgets.QPushButton(Clr_settings)
        self.pushButton_ok.setObjectName("pushButton_ok")
        self.gridLayout.addWidget(self.pushButton_ok, 2, 2, 1, 1)

        self.retranslateUi(Clr_settings)
        QtCore.QMetaObject.connectSlotsByName(Clr_settings)

    def retranslateUi(self, Clr_settings):
        _translate = QtCore.QCoreApplication.translate
        Clr_settings.setWindowTitle(_translate("Clr_settings", "Advanced settings for cyclical learning rates",None))
        self.label_stepSize.setText(_translate("Clr_settings", "step_size",None))
        self.label_gamma.setText(_translate("Clr_settings", "gamma",None))
        self.pushButton_cancel.setText(_translate("Clr_settings", "Close",None))
        self.pushButton_ok.setText(_translate("Clr_settings", "OK",None))




class Ui_Form_expt_optim(QtWidgets.QWidget):
    def setupUi(self, Form_expt_optim):
        Form_expt_optim.setObjectName("Form_expt_optim")
        Form_expt_optim.resize(648, 356)
        self.gridLayout_2 = QtWidgets.QGridLayout(Form_expt_optim)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.groupBox_expt_optim = QtWidgets.QGroupBox(Form_expt_optim)
        self.groupBox_expt_optim.setCheckable(False)
        self.groupBox_expt_optim.setChecked(False)
        self.groupBox_expt_optim.setObjectName("groupBox_expt_optim")
        self.gridLayout_47 = QtWidgets.QGridLayout(self.groupBox_expt_optim)
        self.gridLayout_47.setObjectName("gridLayout_47")
        self.scrollArea_expt_optim = QtWidgets.QScrollArea(self.groupBox_expt_optim)
        self.scrollArea_expt_optim.setWidgetResizable(True)
        self.scrollArea_expt_optim.setObjectName("scrollArea_expt_optim")
        self.scrollAreaWidgetContents_expt_optim = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_expt_optim.setGeometry(QtCore.QRect(0, 0, 600, 257))
        self.scrollAreaWidgetContents_expt_optim.setObjectName("scrollAreaWidgetContents_expt_optim")
        self.gridLayout = QtWidgets.QGridLayout(self.scrollAreaWidgetContents_expt_optim)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.scrollAreaWidgetContents_expt_optim)
        self.label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.label_learningRate = QtWidgets.QLabel(self.scrollAreaWidgetContents_expt_optim)
        self.label_learningRate.setAlignment(QtCore.Qt.AlignCenter)
        self.label_learningRate.setObjectName("label_learningRate")
        self.gridLayout.addWidget(self.label_learningRate, 0, 1, 1, 1)
        self.radioButton_adam = QtWidgets.QRadioButton(self.scrollAreaWidgetContents_expt_optim)
        self.radioButton_adam.setChecked(False)
        self.radioButton_adam.setObjectName("radioButton_adam")
        self.gridLayout.addWidget(self.radioButton_adam, 1, 0, 1, 1)
        self.doubleSpinBox_lr_adam = QtWidgets.QDoubleSpinBox(self.scrollAreaWidgetContents_expt_optim)
        self.doubleSpinBox_lr_adam.setDecimals(6)
        self.doubleSpinBox_lr_adam.setSingleStep(0.0001)
        #self.doubleSpinBox_lr_adam.setProperty("value", 0.001)
        self.doubleSpinBox_lr_adam.setEnabled(False)
        self.doubleSpinBox_lr_adam.setObjectName("doubleSpinBox_lr_adam")
        self.gridLayout.addWidget(self.doubleSpinBox_lr_adam, 1, 1, 1, 1)
        self.label_adam_beta1 = QtWidgets.QLabel(self.scrollAreaWidgetContents_expt_optim)
        self.label_adam_beta1.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_adam_beta1.setObjectName("label_adam_beta1")
        self.gridLayout.addWidget(self.label_adam_beta1, 1, 2, 1, 1)
        self.doubleSpinBox_adam_beta1 = QtWidgets.QDoubleSpinBox(self.scrollAreaWidgetContents_expt_optim)
        self.doubleSpinBox_adam_beta1.setDecimals(3)
        self.doubleSpinBox_adam_beta1.setSingleStep(0.01)
        #self.doubleSpinBox_adam_beta1.setProperty("value", 0.9)
        self.doubleSpinBox_adam_beta1.setEnabled(False)
        self.doubleSpinBox_adam_beta1.setObjectName("doubleSpinBox_adam_beta1")
        self.gridLayout.addWidget(self.doubleSpinBox_adam_beta1, 1, 3, 1, 1)
        self.label_adam_beta2 = QtWidgets.QLabel(self.scrollAreaWidgetContents_expt_optim)
        self.label_adam_beta2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_adam_beta2.setObjectName("label_adam_beta2")
        self.gridLayout.addWidget(self.label_adam_beta2, 1, 4, 1, 1)
        self.doubleSpinBox_adam_beta2 = QtWidgets.QDoubleSpinBox(self.scrollAreaWidgetContents_expt_optim)
        self.doubleSpinBox_adam_beta2.setDecimals(3)
        self.doubleSpinBox_adam_beta2.setSingleStep(0.01)
        #self.doubleSpinBox_adam_beta2.setProperty("value", 0.999)
        self.doubleSpinBox_adam_beta2.setEnabled(False)
        self.doubleSpinBox_adam_beta2.setObjectName("doubleSpinBox_adam_beta2")
        self.gridLayout.addWidget(self.doubleSpinBox_adam_beta2, 1, 5, 1, 1)
        self.checkBox_adam_amsgrad = QtWidgets.QCheckBox(self.scrollAreaWidgetContents_expt_optim)
        self.checkBox_adam_amsgrad.setEnabled(False)
        self.checkBox_adam_amsgrad.setObjectName("checkBox_adam_amsgrad")
        self.gridLayout.addWidget(self.checkBox_adam_amsgrad, 1, 6, 1, 1)
        self.radioButton_sgd = QtWidgets.QRadioButton(self.scrollAreaWidgetContents_expt_optim)
        self.radioButton_sgd.setObjectName("radioButton_sgd")
        self.gridLayout.addWidget(self.radioButton_sgd, 2, 0, 1, 1)
        self.doubleSpinBox_lr_sgd = QtWidgets.QDoubleSpinBox(self.scrollAreaWidgetContents_expt_optim)
        self.doubleSpinBox_lr_sgd.setEnabled(False)
        self.doubleSpinBox_lr_sgd.setDecimals(6)
        self.doubleSpinBox_lr_sgd.setSingleStep(0.0001)
        #self.doubleSpinBox_lr_sgd.setProperty("value", 0.01)
        self.doubleSpinBox_lr_sgd.setObjectName("doubleSpinBox_lr_sgd")
        self.gridLayout.addWidget(self.doubleSpinBox_lr_sgd, 2, 1, 1, 1)
        self.label_sgd_momentum = QtWidgets.QLabel(self.scrollAreaWidgetContents_expt_optim)
        self.label_sgd_momentum.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_sgd_momentum.setObjectName("label_sgd_momentum")
        self.gridLayout.addWidget(self.label_sgd_momentum, 2, 2, 1, 1)
        self.doubleSpinBox_sgd_momentum = QtWidgets.QDoubleSpinBox(self.scrollAreaWidgetContents_expt_optim)
        self.doubleSpinBox_sgd_momentum.setEnabled(False)
        self.doubleSpinBox_sgd_momentum.setDecimals(3)
        self.doubleSpinBox_sgd_momentum.setSingleStep(0.01)
        self.doubleSpinBox_sgd_momentum.setObjectName("doubleSpinBox_sgd_momentum")
        self.gridLayout.addWidget(self.doubleSpinBox_sgd_momentum, 2, 3, 1, 1)
        self.checkBox_sgd_nesterov = QtWidgets.QCheckBox(self.scrollAreaWidgetContents_expt_optim)
        self.checkBox_sgd_nesterov.setEnabled(False)
        self.checkBox_sgd_nesterov.setObjectName("checkBox_sgd_nesterov")
        self.gridLayout.addWidget(self.checkBox_sgd_nesterov, 2, 4, 1, 2)
        self.radioButton_rms = QtWidgets.QRadioButton(self.scrollAreaWidgetContents_expt_optim)
        self.radioButton_rms.setObjectName("radioButton_rms")
        self.gridLayout.addWidget(self.radioButton_rms, 3, 0, 1, 1)
        self.doubleSpinBox_lr_rmsprop = QtWidgets.QDoubleSpinBox(self.scrollAreaWidgetContents_expt_optim)
        self.doubleSpinBox_lr_rmsprop.setEnabled(False)
        self.doubleSpinBox_lr_rmsprop.setDecimals(6)
        self.doubleSpinBox_lr_rmsprop.setSingleStep(0.0001)
        #self.doubleSpinBox_lr_rmsprop.setProperty("value", 0.001)
        self.doubleSpinBox_lr_rmsprop.setObjectName("doubleSpinBox_lr_rmsprop")
        self.gridLayout.addWidget(self.doubleSpinBox_lr_rmsprop, 3, 1, 1, 1)
        self.label_rms_rho = QtWidgets.QLabel(self.scrollAreaWidgetContents_expt_optim)
        self.label_rms_rho.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_rms_rho.setObjectName("label_rms_rho")
        self.gridLayout.addWidget(self.label_rms_rho, 3, 2, 1, 1)
        self.doubleSpinBox_rms_rho = QtWidgets.QDoubleSpinBox(self.scrollAreaWidgetContents_expt_optim)
        self.doubleSpinBox_rms_rho.setEnabled(False)
        self.doubleSpinBox_rms_rho.setDecimals(3)
        self.doubleSpinBox_rms_rho.setSingleStep(0.01)
        #self.doubleSpinBox_rms_rho.setProperty("value", 0.9)
        self.doubleSpinBox_rms_rho.setObjectName("doubleSpinBox_rms_rho")
        self.gridLayout.addWidget(self.doubleSpinBox_rms_rho, 3, 3, 1, 1)
        self.radioButton_adagrad = QtWidgets.QRadioButton(self.scrollAreaWidgetContents_expt_optim)
        self.radioButton_adagrad.setObjectName("radioButton_adagrad")
        self.gridLayout.addWidget(self.radioButton_adagrad, 4, 0, 1, 1)
        self.doubleSpinBox_lr_adagrad = QtWidgets.QDoubleSpinBox(self.scrollAreaWidgetContents_expt_optim)
        self.doubleSpinBox_lr_adagrad.setEnabled(False)
        self.doubleSpinBox_lr_adagrad.setDecimals(6)
        self.doubleSpinBox_lr_adagrad.setSingleStep(0.0001)
        #self.doubleSpinBox_lr_adagrad.setProperty("value", 0.01)
        self.doubleSpinBox_lr_adagrad.setObjectName("doubleSpinBox_lr_adagrad")
        self.gridLayout.addWidget(self.doubleSpinBox_lr_adagrad, 4, 1, 1, 1)
        self.radioButton_adadelta = QtWidgets.QRadioButton(self.scrollAreaWidgetContents_expt_optim)
        self.radioButton_adadelta.setObjectName("radioButton_adadelta")
        self.gridLayout.addWidget(self.radioButton_adadelta, 5, 0, 1, 1)
        self.doubleSpinBox_lr_adadelta = QtWidgets.QDoubleSpinBox(self.scrollAreaWidgetContents_expt_optim)
        self.doubleSpinBox_lr_adadelta.setEnabled(False)
        self.doubleSpinBox_lr_adadelta.setDecimals(6)
        self.doubleSpinBox_lr_adadelta.setSingleStep(0.0001)
        #self.doubleSpinBox_lr_adadelta.setProperty("value", 1.0)
        self.doubleSpinBox_lr_adadelta.setObjectName("doubleSpinBox_lr_adadelta")
        self.gridLayout.addWidget(self.doubleSpinBox_lr_adadelta, 5, 1, 1, 1)
        self.label_adadelta_rho = QtWidgets.QLabel(self.scrollAreaWidgetContents_expt_optim)
        self.label_adadelta_rho.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_adadelta_rho.setObjectName("label_adadelta_rho")
        self.gridLayout.addWidget(self.label_adadelta_rho, 5, 2, 1, 1)
        self.doubleSpinBox_adadelta_rho = QtWidgets.QDoubleSpinBox(self.scrollAreaWidgetContents_expt_optim)
        self.doubleSpinBox_adadelta_rho.setEnabled(False)
        self.doubleSpinBox_adadelta_rho.setDecimals(3)
        self.doubleSpinBox_adadelta_rho.setSingleStep(0.01)
        #self.doubleSpinBox_adadelta_rho.setProperty("value", 0.95)
        self.doubleSpinBox_adadelta_rho.setObjectName("doubleSpinBox_adadelta_rho")
        self.gridLayout.addWidget(self.doubleSpinBox_adadelta_rho, 5, 3, 1, 1)
        self.radioButton_adamax = QtWidgets.QRadioButton(self.scrollAreaWidgetContents_expt_optim)
        self.radioButton_adamax.setObjectName("radioButton_adamax")
        self.gridLayout.addWidget(self.radioButton_adamax, 6, 0, 1, 1)
        self.doubleSpinBox_lr_adamax = QtWidgets.QDoubleSpinBox(self.scrollAreaWidgetContents_expt_optim)
        self.doubleSpinBox_lr_adamax.setEnabled(False)
        self.doubleSpinBox_lr_adamax.setDecimals(6)
        self.doubleSpinBox_lr_adamax.setSingleStep(0.0001)
        self.doubleSpinBox_lr_adamax.setProperty("value", 0.002)
        self.doubleSpinBox_lr_adamax.setObjectName("doubleSpinBox_lr_adamax")
        self.gridLayout.addWidget(self.doubleSpinBox_lr_adamax, 6, 1, 1, 1)
        self.label_adamax_beta1 = QtWidgets.QLabel(self.scrollAreaWidgetContents_expt_optim)
        self.label_adamax_beta1.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_adamax_beta1.setObjectName("label_adamax_beta1")
        self.gridLayout.addWidget(self.label_adamax_beta1, 6, 2, 1, 1)
        self.doubleSpinBox_adamax_beta1 = QtWidgets.QDoubleSpinBox(self.scrollAreaWidgetContents_expt_optim)
        self.doubleSpinBox_adamax_beta1.setEnabled(False)
        self.doubleSpinBox_adamax_beta1.setDecimals(3)
        self.doubleSpinBox_adamax_beta1.setSingleStep(0.01)
        #self.doubleSpinBox_adamax_beta1.setProperty("value", 0.9)
        self.doubleSpinBox_adamax_beta1.setObjectName("doubleSpinBox_adamax_beta1")
        self.gridLayout.addWidget(self.doubleSpinBox_adamax_beta1, 6, 3, 1, 1)
        self.label_adamax_beta2 = QtWidgets.QLabel(self.scrollAreaWidgetContents_expt_optim)
        self.label_adamax_beta2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_adamax_beta2.setObjectName("label_adamax_beta2")
        self.gridLayout.addWidget(self.label_adamax_beta2, 6, 4, 1, 1)
        self.doubleSpinBox_adamax_beta2 = QtWidgets.QDoubleSpinBox(self.scrollAreaWidgetContents_expt_optim)
        self.doubleSpinBox_adamax_beta2.setEnabled(False)
        self.doubleSpinBox_adamax_beta2.setDecimals(3)
        self.doubleSpinBox_adamax_beta2.setSingleStep(0.01)
        #self.doubleSpinBox_adamax_beta2.setProperty("value", 0.999)
        self.doubleSpinBox_adamax_beta2.setObjectName("doubleSpinBox_adamax_beta2")
        self.gridLayout.addWidget(self.doubleSpinBox_adamax_beta2, 6, 5, 1, 1)
        self.radioButton_nadam = QtWidgets.QRadioButton(self.scrollAreaWidgetContents_expt_optim)
        self.radioButton_nadam.setObjectName("radioButton_nadam")
        self.gridLayout.addWidget(self.radioButton_nadam, 7, 0, 1, 1)
        self.doubleSpinBox_lr_nadam = QtWidgets.QDoubleSpinBox(self.scrollAreaWidgetContents_expt_optim)
        self.doubleSpinBox_lr_nadam.setEnabled(False)
        self.doubleSpinBox_lr_nadam.setDecimals(6)
        self.doubleSpinBox_lr_nadam.setSingleStep(0.0001)
        #self.doubleSpinBox_lr_nadam.setProperty("value", 0.002)
        self.doubleSpinBox_lr_nadam.setObjectName("doubleSpinBox_lr_nadam")
        self.gridLayout.addWidget(self.doubleSpinBox_lr_nadam, 7, 1, 1, 1)
        self.label_nadam_beta1 = QtWidgets.QLabel(self.scrollAreaWidgetContents_expt_optim)
        self.label_nadam_beta1.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_nadam_beta1.setObjectName("label_nadam_beta1")
        self.gridLayout.addWidget(self.label_nadam_beta1, 7, 2, 1, 1)
        self.doubleSpinBox_nadam_beta1 = QtWidgets.QDoubleSpinBox(self.scrollAreaWidgetContents_expt_optim)
        self.doubleSpinBox_nadam_beta1.setEnabled(False)
        self.doubleSpinBox_nadam_beta1.setDecimals(3)
        self.doubleSpinBox_nadam_beta1.setSingleStep(0.01)
        #self.doubleSpinBox_nadam_beta1.setProperty("value", 0.9)
        self.doubleSpinBox_nadam_beta1.setObjectName("doubleSpinBox_nadam_beta1")
        self.gridLayout.addWidget(self.doubleSpinBox_nadam_beta1, 7, 3, 1, 1)
        self.label_nadam_beta2 = QtWidgets.QLabel(self.scrollAreaWidgetContents_expt_optim)
        self.label_nadam_beta2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_nadam_beta2.setObjectName("label_nadam_beta2")
        self.gridLayout.addWidget(self.label_nadam_beta2, 7, 4, 1, 1)
        self.doubleSpinBox_nadam_beta2 = QtWidgets.QDoubleSpinBox(self.scrollAreaWidgetContents_expt_optim)
        self.doubleSpinBox_nadam_beta2.setEnabled(False)
        self.doubleSpinBox_nadam_beta2.setDecimals(3)
        self.doubleSpinBox_nadam_beta2.setSingleStep(0.01)
        #self.doubleSpinBox_nadam_beta2.setProperty("value", 0.999)
        self.doubleSpinBox_nadam_beta2.setObjectName("doubleSpinBox_nadam_beta2")
        self.gridLayout.addWidget(self.doubleSpinBox_nadam_beta2, 7, 5, 1, 1)
        self.scrollArea_expt_optim.setWidget(self.scrollAreaWidgetContents_expt_optim)
        self.gridLayout_47.addWidget(self.scrollArea_expt_optim, 0, 1, 1, 1)
        self.gridLayout_2.addWidget(self.groupBox_expt_optim, 0, 0, 1, 4)
        spacerItem = QtWidgets.QSpacerItem(323, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem, 1, 0, 1, 1)
        self.pushButton_cancel = QtWidgets.QPushButton(Form_expt_optim)
        self.pushButton_cancel.setObjectName("pushButton_cancel")
        self.gridLayout_2.addWidget(self.pushButton_cancel, 1, 1, 1, 1)
        self.pushButton_reset = QtWidgets.QPushButton(Form_expt_optim)
        self.pushButton_reset.setObjectName("pushButton_reset")
        self.gridLayout_2.addWidget(self.pushButton_reset, 1, 2, 1, 1)
        self.pushButton_ok = QtWidgets.QPushButton(Form_expt_optim)
        self.pushButton_ok.setObjectName("pushButton_ok")
        self.gridLayout_2.addWidget(self.pushButton_ok, 1, 3, 1, 1)

        
        self.radioButton_adam.toggled['bool'].connect(self.doubleSpinBox_lr_adam.setEnabled)
        self.radioButton_adam.toggled['bool'].connect(self.doubleSpinBox_adam_beta1.setEnabled)
        self.radioButton_adam.toggled['bool'].connect(self.doubleSpinBox_adam_beta2.setEnabled)
        self.radioButton_adam.toggled['bool'].connect(self.checkBox_adam_amsgrad.setEnabled)
        self.radioButton_sgd.toggled['bool'].connect(self.doubleSpinBox_lr_sgd.setEnabled)
        self.radioButton_sgd.toggled['bool'].connect(self.doubleSpinBox_sgd_momentum.setEnabled)
        self.radioButton_sgd.toggled['bool'].connect(self.checkBox_sgd_nesterov.setEnabled)
        self.radioButton_rms.toggled['bool'].connect(self.doubleSpinBox_lr_rmsprop.setEnabled)
        self.radioButton_rms.toggled['bool'].connect(self.doubleSpinBox_rms_rho.setEnabled)
        self.radioButton_adagrad.toggled['bool'].connect(self.doubleSpinBox_lr_adagrad.setEnabled)
        self.radioButton_adadelta.toggled['bool'].connect(self.doubleSpinBox_lr_adadelta.setEnabled)
        self.radioButton_adadelta.toggled['bool'].connect(self.doubleSpinBox_adadelta_rho.setEnabled)
        self.radioButton_adamax.toggled['bool'].connect(self.doubleSpinBox_lr_adamax.setEnabled)
        self.radioButton_adamax.toggled['bool'].connect(self.doubleSpinBox_adamax_beta1.setEnabled)
        self.radioButton_adamax.toggled['bool'].connect(self.doubleSpinBox_adamax_beta2.setEnabled)
        self.radioButton_nadam.toggled['bool'].connect(self.doubleSpinBox_lr_nadam.setEnabled)
        self.radioButton_nadam.toggled['bool'].connect(self.doubleSpinBox_nadam_beta1.setEnabled)
        self.radioButton_nadam.toggled['bool'].connect(self.doubleSpinBox_nadam_beta2.setEnabled)

        self.retranslateUi(Form_expt_optim)    
        QtCore.QMetaObject.connectSlotsByName(Form_expt_optim)

    def retranslateUi(self, Form_expt_optim):
        _translate = QtCore.QCoreApplication.translate
        Form_expt_optim.setWindowTitle(_translate("Form_expt_optim", "Change optimizer settings"))
        self.groupBox_expt_optim.setTitle(_translate("Form_expt_optim", "Optimizer Settings"))
        self.label.setText(_translate("Form_expt_optim", "Optimizer"))
        self.label_learningRate.setText(_translate("Form_expt_optim", "Learning Rate"))
        self.radioButton_adam.setToolTip(_translate("Form_expt_optim", "Adam optimizer.\n"
"\n"
"Default parameters follow those provided in the original paper."))
        self.radioButton_adam.setText(_translate("Form_expt_optim", "Adam"))
        self.label_adam_beta1.setText(_translate("Form_expt_optim", "beta_1"))
        self.label_adam_beta2.setText(_translate("Form_expt_optim", "beta_2"))
        self.checkBox_adam_amsgrad.setText(_translate("Form_expt_optim", "amsgrad"))
        self.radioButton_sgd.setToolTip(_translate("Form_expt_optim", "Stochastic gradient descent optimizer.\n"
"\n"
"Includes support for momentum, learning rate decay, and Nesterov momentum."))
        self.radioButton_sgd.setText(_translate("Form_expt_optim", "SGD"))
        self.label_sgd_momentum.setText(_translate("Form_expt_optim", "Momentum"))
        self.checkBox_sgd_nesterov.setText(_translate("Form_expt_optim", "Nesterov"))
        self.radioButton_rms.setToolTip(_translate("Form_expt_optim", "RMSProp optimizer.\n"
"\n"
"It is recommended to leave the parameters of this optimizer at their default values (except the learning rate, which can be freely tuned)."))
        self.radioButton_rms.setText(_translate("Form_expt_optim", "RMSprop"))
        self.label_rms_rho.setText(_translate("Form_expt_optim", "Rho"))
        self.radioButton_adagrad.setToolTip(_translate("Form_expt_optim", "Adagrad optimizer.\n"
"\n"
"Adagrad is an optimizer with parameter-specific learning rates, which are adapted relative to how frequently a parameter gets updated during training. The more updates a parameter receives, the smaller the learning rate.\n"
"\n"
"It is recommended to leave the parameters of this optimizer at their default values."))
        self.radioButton_adagrad.setText(_translate("Form_expt_optim", "Adagrad"))
        self.radioButton_adadelta.setToolTip(_translate("Form_expt_optim", "Adadelta optimizer.\n"
"\n"
"Adadelta is a more robust extension of Adagrad that adapts learning rates based on a moving window of gradient updates, instead of accumulating all past gradients. This way, Adadelta continues learning even when many updates have been done. Compared to Adagrad, in the original version of Adadelta you don\'t have to set an initial learning rate. In this version, initial learning rate and decay factor can be set, as in most other Keras optimizers.\n"
"\n"
"It is recommended to leave the parameters of this optimizer at their default values."))
        self.radioButton_adadelta.setText(_translate("Form_expt_optim", "Adadelta"))
        self.label_adadelta_rho.setText(_translate("Form_expt_optim", "Rho"))
        self.radioButton_adamax.setToolTip(_translate("Form_expt_optim", "Adamax optimizer from Adam paper\'s Section 7.\n"
"\n"
"It is a variant of Adam based on the infinity norm. Default parameters follow those provided in the paper."))
        self.radioButton_adamax.setText(_translate("Form_expt_optim", "Adamax"))
        self.label_adamax_beta1.setText(_translate("Form_expt_optim", "beta_1"))
        self.label_adamax_beta2.setText(_translate("Form_expt_optim", "beta_2"))
        self.radioButton_nadam.setToolTip(_translate("Form_expt_optim", "Nesterov Adam optimizer.\n"
"\n"
"Much like Adam is essentially RMSprop with momentum, Nadam is RMSprop with Nesterov momentum.\n"
"\n"
"Default parameters follow those provided in the paper. It is recommended to leave the parameters of this optimizer at their default values."))
        self.radioButton_nadam.setText(_translate("Form_expt_optim", "Nadam"))
        self.label_nadam_beta1.setText(_translate("Form_expt_optim", "beta_1"))
        self.label_nadam_beta2.setText(_translate("Form_expt_optim", "beta_2"))
        self.pushButton_cancel.setText(_translate("Form_expt_optim", "Cancel"))
        self.pushButton_reset.setText(_translate("Form_expt_optim", "Reset"))
        self.pushButton_ok.setText(_translate("Form_expt_optim", "OK"))


def message(msg_text,msg_type="Error"):
    #There was an error!
    msg = QtWidgets.QMessageBox()
    if msg_type=="Error":
        msg.setIcon(QtWidgets.QMessageBox.Critical)       
    elif msg_type=="Information":
        msg.setIcon(QtWidgets.QMessageBox.Information)       
    elif msg_type=="Question":
        msg.setIcon(QtWidgets.QMessageBox.Question)       
    elif msg_type=="Warning":
        msg.setIcon(QtWidgets.QMessageBox.Warning)       
    msg.setText(str(msg_text))
    msg.setWindowTitle(msg_type)
    msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
    msg.exec_()
    pass


def load_hyper_params(ui_item,para):
    if para["new_model"].iloc[-1]==True:
        ui_item.radioButton_NewModel.setChecked(True)
        prop = str(para["Chosen Model"].iloc[-1])
        index = ui_item.comboBox_ModelSelection.findText(prop, QtCore.Qt.MatchFixedString)
        if index >= 0:
            ui_item.comboBox_ModelSelection.setCurrentIndex(index)
    elif para["loadrestart_model"].iloc[-1]==True:
        ui_item.radioButton_NewModel.setChecked(True)
        prop = str(para["Continued_Fitting_From"])
        ui_item.lineEdit_LoadModelPath.setText(prop)
    elif para["loadcontinue_model"].iloc[-1]==True:
        ui_item.radioButton_LoadContinueModel.setChecked(True)
        prop = str(para["Continued_Fitting_From"].iloc[-1])
        ui_item.lineEdit_LoadModelPath.setText(prop)
    if "Input image size" in para.keys():
        prop = para["Input image size"].iloc[-1]
    elif "Input image crop" in para.keys():
        prop = para["Input image crop"].iloc[-1]
    else:
        prop = 32
        print("Cound not find parameter for 'Input image size' in the meta file")
    ui_item.spinBox_imagecrop.setValue(prop)
    try:
        prop = para["Color Mode"].iloc[-1]
        index = ui_item.comboBox_GrayOrRGB.findText(prop, QtCore.Qt.MatchFixedString)
        if index >= 0:
            ui_item.comboBox_GrayOrRGB.setCurrentIndex(index)
    except Exception as e:
        message(e)
    try:
        prop = int(para["Zoom order"].iloc[-1])
        ui_item.comboBox_zoomOrder.setCurrentIndex(prop)
    except Exception as e:
        message(e)
    try:
        prop = str(para["Normalization"].iloc[-1])
        index = ui_item.comboBox_Normalization.findText(prop, QtCore.Qt.MatchFixedString)
        if index >= 0:
            ui_item.comboBox_Normalization.setCurrentIndex(index)
    except Exception as e:
        message(e)
    try:
        prop = int(para["Nr. epochs"].iloc[-1])
        ui_item.spinBox_NrEpochs.setValue(prop)
    except Exception as e:
        message(e)
    try:
        prop = int(para["Keras refresh after nr. epochs"].iloc[-1])
        ui_item.spinBox_RefreshAfterEpochs.setValue(prop)
        prop = bool(para["Horz. flip"].iloc[-1])
        ui_item.checkBox_HorizFlip.setChecked(prop)
        prop = bool(para["Vert. flip"].iloc[-1])
        ui_item.checkBox_HorizFlip.setChecked(prop)
        prop = str(para["rotation"].iloc[-1])
        ui_item.lineEdit_Rotation.setText(prop)
        prop = str(para["width_shift"].iloc[-1])
        ui_item.lineEdit_widthShift.setText(prop)
        prop = str(para["height_shift"].iloc[-1])
        ui_item.lineEdit_heightShift.setText(prop)
        prop = str(para["zoom"].iloc[-1])
        ui_item.lineEdit_zoomRange.setText(prop)
        prop = str(para["shear"].iloc[-1])
        ui_item.lineEdit_shearRange.setText(prop)
    except Exception as e:
        message(e)
    try:    
        prop = int(para["Keras refresh after nr. epochs"].iloc[-1])
        ui_item.spinBox_RefreshAfterNrEpochs.setValue(prop)
        prop = int(para["Brightness add. lower"].iloc[-1])
        ui_item.spinBox_PlusLower.setValue(prop)
        prop = int(para["Brightness add. upper"].iloc[-1])
        ui_item.spinBox_PlusUpper.setValue(prop)
        prop = float(para["Brightness mult. lower"].iloc[-1])
        ui_item.doubleSpinBox_MultLower.setValue(prop)
        prop = float(para["Brightness mult. upper"].iloc[-1])
        ui_item.doubleSpinBox_MultUpper.setValue(prop)
        prop = float(para["Gaussnoise Mean"].iloc[-1])
        ui_item.doubleSpinBox_GaussianNoiseMean.setValue(prop)
        prop = float(para["Gaussnoise Scale"].iloc[-1])
        ui_item.doubleSpinBox_GaussianNoiseScale.setValue(prop)
    except Exception as e:
        message(e)
    try:
        prop = bool(para["Contrast on"].iloc[-1])
        ui_item.checkBox_contrast.setChecked(prop)
        prop = float(para["Contrast Lower"].iloc[-1])
        ui_item.doubleSpinBox_contrastLower.setValue(prop)
        prop = float(para["Contrast Higher"].iloc[-1])
        ui_item.doubleSpinBox_contrastHigher.setValue(prop)
        prop = bool(para["Saturation on"].iloc[-1])
        ui_item.checkBox_saturation.setChecked(prop)
        prop = float(para["Saturation Lower"].iloc[-1])
        ui_item.doubleSpinBox_saturationLower.setValue(prop)
        prop = float(para["Saturation Higher"].iloc[-1])
        ui_item.doubleSpinBox_saturationHigher.setValue(prop)
        prop = bool(para["Hue on"].iloc[-1])
        ui_item.checkBox_hue.setChecked(prop)
        prop = float(para["Hue delta"].iloc[-1])
        ui_item.doubleSpinBox_hueDelta.setValue(prop)
    except Exception as e:
        message(e)
    try:
        prop = bool(para["Average blur on"].iloc[-1])
        ui_item.checkBox_avgBlur.setChecked(prop)
        prop = int(para["Average blur Lower"].iloc[-1])
        ui_item.spinBox_avgBlurMin.setValue(prop)
        prop = int(para["Average blur Higher"].iloc[-1])
        ui_item.spinBox_avgBlurMax.setValue(prop)
        prop = bool(para["Gauss blur on"].iloc[-1])
        ui_item.checkBox_gaussBlur.setChecked(prop)
        prop = int(para["Gauss blur Lower"].iloc[-1])
        ui_item.spinBox_gaussBlurMin.setValue(prop)
        prop = int(para["Gauss blur Higher"].iloc[-1])
        ui_item.spinBox_gaussBlurMax.setValue(prop)
        prop = bool(para["Motion blur on"].iloc[-1])
        ui_item.checkBox_motionBlur.setChecked(prop)
        prop = str(para["Motion blur Kernel"].iloc[-1])
        ui_item.lineEdit_motionBlurKernel.setText(prop)
        prop = str(para["Motion blur Angle"].iloc[-1])
        ui_item.lineEdit_motionBlurAngle.setText(prop)
    except Exception as e:
        message(e)
    try:
        prop = bool(para["expert_mode"].iloc[-1])
        ui_item.groupBox_expertMode.setChecked(prop)
    except Exception as e:
        message(e)
    try:
        prop = str(para["optimizer_settings"].iloc[-1])
        prop = eval(prop)
        ui_item.optimizer_settings["doubleSpinBox_lr_sgd"] = prop["doubleSpinBox_lr_sgd"]
        ui_item.optimizer_settings["doubleSpinBox_sgd_momentum"] = prop["doubleSpinBox_sgd_momentum"]
        ui_item.optimizer_settings["checkBox_sgd_nesterov"] = prop["checkBox_sgd_nesterov"]
    
        ui_item.optimizer_settings["doubleSpinBox_lr_rmsprop"] = prop["doubleSpinBox_lr_rmsprop"]
        ui_item.optimizer_settings["doubleSpinBox_rms_rho"] = prop["doubleSpinBox_rms_rho"]
    
        ui_item.optimizer_settings["doubleSpinBox_lr_adam"] = prop["doubleSpinBox_lr_adam"]
        ui_item.optimizer_settings["doubleSpinBox_adam_beta1"] = prop["doubleSpinBox_adam_beta1"]
        ui_item.optimizer_settings["doubleSpinBox_adam_beta2"] = prop["doubleSpinBox_adam_beta2"]
        ui_item.optimizer_settings["checkBox_adam_amsgrad"] = prop["checkBox_adam_amsgrad"]
    
        ui_item.optimizer_settings["doubleSpinBox_lr_adadelta"] = prop["doubleSpinBox_lr_adadelta"]
        ui_item.optimizer_settings["doubleSpinBox_adadelta_rho"] = prop["doubleSpinBox_adadelta_rho"]
    
        ui_item.optimizer_settings["doubleSpinBox_lr_nadam"] = prop["doubleSpinBox_lr_nadam"]
        ui_item.optimizer_settings["doubleSpinBox_nadam_beta1"] = prop["doubleSpinBox_nadam_beta1"]
        ui_item.optimizer_settings["doubleSpinBox_nadam_beta2"] = prop["doubleSpinBox_nadam_beta2"]
    
        ui_item.optimizer_settings["doubleSpinBox_lr_adagrad"] = prop["doubleSpinBox_lr_adagrad"]
    
        ui_item.optimizer_settings["doubleSpinBox_lr_adamax"] = prop["doubleSpinBox_lr_adamax"]
        ui_item.optimizer_settings["doubleSpinBox_adamax_beta1"] = prop["doubleSpinBox_adamax_beta1"]
        ui_item.optimizer_settings["doubleSpinBox_adamax_beta2"] = prop["doubleSpinBox_adamax_beta2"]
    except Exception as e:
        message(e)
    try:
        prop = bool(para["optimizer_expert_on"].iloc[-1])
        ui_item.checkBox_optimizer.setChecked(prop)
        prop = str(para["optimizer_expert"].iloc[-1])
        index = ui_item.comboBox_optimizer.findText(prop, QtCore.Qt.MatchFixedString)
        if index >= 0:
            ui_item.comboBox_optimizer.setCurrentIndex(index)
    except Exception as e:
        message(e)
    try:
        prop = int(para["batchSize_expert"].iloc[-1])
        ui_item.spinBox_batchSize.setValue(prop)
        prop = int(para["epochs_expert"].iloc[-1])
        ui_item.spinBox_epochs.setValue(prop)
    except Exception as e:
        message(e)
    try:
        prop = bool(para["learning_rate_expert_on"].iloc[-1])
        ui_item.groupBox_learningRate.setChecked(prop)
    except Exception as e:
        message(e)
    try:
        prop = bool(para["learning_rate_const_on"].iloc[-1])
        ui_item.radioButton_LrConst.setChecked(prop)
        prop = float(para["learning_rate_const"].iloc[-1])
        ui_item.doubleSpinBox_learningRate.setValue(prop)
    except Exception as e:
        message(e)
    try:
        prop = bool(para["learning_rate_cycLR_on"].iloc[-1])
        ui_item.radioButton_LrCycl.setChecked(prop)
        prop = str(para["cycLrMin"].iloc[-1])
        ui_item.lineEdit_cycLrMin.setText(prop)
        prop = str(para["cycLrMax"].iloc[-1])
        ui_item.lineEdit_cycLrMax.setText(prop)
        prop = str(para["cycLrMethod"].iloc[-1])
        index = ui_item.comboBox_cycLrMethod.findText(prop, QtCore.Qt.MatchFixedString)
        if index >= 0:
            ui_item.comboBox_cycLrMethod.setCurrentIndex(index)
        prop = str(para["clr_settings"].iloc[-1])
        prop = eval(prop)
        ui_item.clr_settings["step_size"] = prop["step_size"]
        ui_item.clr_settings["gamma"] = prop["gamma"]
    except Exception as e:
        message(e)
    try:
        prop = bool(para["learning_rate_expo_on"].iloc[-1])
        ui_item.radioButton_LrExpo.setChecked(prop)
        prop = float(para["expDecInitLr"].iloc[-1])
        ui_item.doubleSpinBox_expDecInitLr.setValue(prop)
        prop = int(para["expDecSteps"].iloc[-1])
        ui_item.spinBox_expDecSteps.setValue(prop)
        prop = float(para["expDecRate"].iloc[-1])
        ui_item.doubleSpinBox_expDecRate.setValue(prop)
    except Exception as e:
        message(e)
    try:
        prop = bool(para["loss_expert_on"].iloc[-1])
        ui_item.checkBox_expt_loss.setChecked(prop)
        prop = str(para["loss_expert"].iloc[-1])
        index = ui_item.comboBox_expt_loss.findText(prop, QtCore.Qt.MatchFixedString)
        if index >= 0:
            ui_item.comboBox_expt_loss.setCurrentIndex(index)
    except Exception as e:
        message(e)
    try:
        prop = str(para["paddingMode"].iloc[-1])
        index = ui_item.comboBox_paddingMode.findText(prop, QtCore.Qt.MatchFixedString)
        if index >= 0:
            ui_item.comboBox_paddingMode.setCurrentIndex(index)
    except Exception as e:
        message(e)
    try:
        prop = bool(para["train_last_layers"].iloc[-1])
        ui_item.checkBox_trainLastNOnly.setChecked(prop)
        prop = int(para["train_last_layers_n"].iloc[-1])
        ui_item.spinBox_trainLastNOnly.setValue(prop)
        prop = bool(para["train_dense_layers"].iloc[-1])
        ui_item.checkBox_trainDenseOnly.setChecked(prop)
    except Exception as e:
        message(e)
    try:
        prop = bool(para["dropout_expert_on"].iloc[-1])
        ui_item.checkBox_dropout.setChecked(prop)
        prop = str(para["dropout_expert"].iloc[-1])
        if prop!="()":
            ui_item.lineEdit_dropout.setText(prop[1:-1])
    except Exception as e:
        message(e)
    try:
        prop = bool(para["lossW_expert_on"].iloc[-1])
        ui_item.checkBox_lossW.setChecked(prop)
        prop = str(para["lossW_expert"].iloc[-1])
        if prop!="nan":
            ui_item.lineEdit_lossW.setText(prop)
    except Exception as e:
        message(e)
    try:
        prop = str(para["metrics"].iloc[-1])
        if "accuracy" in prop.lower():
            ui_item.checkBox_expertAccuracy.setChecked(True)
        if "f1" in prop.lower():
            ui_item.checkBox_expertF1.setChecked(True)
        if "precision" in prop.lower():
            ui_item.checkBox_expertPrecision.setChecked(True)
        if "recall" in prop.lower():
            ui_item.checkBox_expertRecall.setChecked(True)
    except Exception as e:
        message(e)


def get_hyper_params(Para_dict,ui_item):
    Para_dict["Modelname"]=str(ui_item.lineEdit_modelname.text()),
    Para_dict["Chosen Model"]=str(ui_item.comboBox_ModelSelection.currentText()),
    Para_dict["new_model"]=ui_item.radioButton_NewModel.isChecked(),
    Para_dict["loadrestart_model"]=ui_item.radioButton_LoadRestartModel.isChecked(),
    Para_dict["loadcontinue_model"]=ui_item.radioButton_LoadContinueModel.isChecked(),
    if ui_item.radioButton_LoadRestartModel.isChecked():
        load_modelname = str(ui_item.lineEdit_LoadModelPath.text())
    elif ui_item.radioButton_LoadContinueModel.isChecked():
        load_modelname = str(ui_item.lineEdit_LoadModelPath.text())
    elif ui_item.radioButton_NewModel.isChecked():
        load_modelname = "" #No model is loaded
    else:
        load_modelname = ""
    Para_dict["Continued_Fitting_From"]=load_modelname,                        
    Para_dict["Input image size"]=int(ui_item.spinBox_imagecrop.value()) ,
    Para_dict["Color Mode"]=str(ui_item.comboBox_GrayOrRGB.currentText()),
    try: Para_dict["Zoom order"]=int(ui_item.comboBox_zoomOrder.currentIndex()), 
    except Exception as e:
        message(e)
    try:
        if ui_item.radioButton_cpu.isChecked():
            gpu_used = False
            deviceSelected = str(ui_item.comboBox_cpu.currentText())
        elif ui_item.radioButton_gpu.isChecked():
            gpu_used = True
            deviceSelected = str(ui_item.comboBox_gpu.currentText())
        gpu_memory = float(ui_item.doubleSpinBox_memory.value())
        Para_dict["Device"]=deviceSelected,
        Para_dict["gpu_used"]=gpu_used,
        Para_dict["gpu_memory"]=gpu_memory,
    except Exception as e:
        message(e)
    Para_dict["Output Nr. classes"]=np.nan,
    norm = str(ui_item.comboBox_Normalization.currentText())
    Para_dict["Normalization"]=norm,
    Para_dict["Nr. epochs"]=int(ui_item.spinBox_NrEpochs.value()),
    try:
        Para_dict["Keras refresh after nr. epochs"]=int(ui_item.spinBox_RefreshAfterEpochs.value()),
        Para_dict["Horz. flip"]= bool(ui_item.checkBox_HorizFlip.isChecked()),
        Para_dict["Vert. flip"]= bool(ui_item.checkBox_VertFlip.isChecked()),
        Para_dict["rotation"]=float(ui_item.lineEdit_Rotation.text()),
        Para_dict["width_shift"]=float(ui_item.lineEdit_widthShift.text()),
        Para_dict["height_shift"]=float(ui_item.lineEdit_heightShift.text()),
        Para_dict["zoom"]=float(ui_item.lineEdit_zoomRange.text()),
        Para_dict["shear"]=float(ui_item.lineEdit_shearRange.text()),
    except Exception as e:
        message(e)
    try:
        Para_dict["Brightness refresh after nr. epochs"]=int(ui_item.spinBox_RefreshAfterNrEpochs.value()),
        Para_dict["Brightness add. lower"]=float(ui_item.spinBox_PlusLower.value()),
        Para_dict["Brightness add. upper"]=float(ui_item.spinBox_PlusUpper.value()),
        Para_dict["Brightness mult. lower"]=float(ui_item.doubleSpinBox_MultLower.value()),  
        Para_dict["Brightness mult. upper"]=float(ui_item.doubleSpinBox_MultUpper.value()),
        Para_dict["Gaussnoise Mean"]=float(ui_item.doubleSpinBox_GaussianNoiseMean.value()),
        Para_dict["Gaussnoise Scale"]=float(ui_item.doubleSpinBox_GaussianNoiseScale.value()),
    except Exception as e:
        message(e)
    try:
        Para_dict["Contrast on"]=bool(ui_item.checkBox_contrast.isChecked()) ,                
        Para_dict["Contrast Lower"]=float(ui_item.doubleSpinBox_contrastLower.value()),
        Para_dict["Contrast Higher"]=float(ui_item.doubleSpinBox_contrastHigher.value()),
        Para_dict["Saturation on"]=bool(ui_item.checkBox_saturation.isChecked()),
        Para_dict["Saturation Lower"]=float(ui_item.doubleSpinBox_saturationLower.value()),
        Para_dict["Saturation Higher"]=float(ui_item.doubleSpinBox_saturationHigher.value()),
        Para_dict["Hue on"]=bool(ui_item.checkBox_hue.isChecked()),                
        Para_dict["Hue delta"]=float(ui_item.doubleSpinBox_hueDelta.value()),                
    except Exception as e:
        message(e)
    try:
        Para_dict["Average blur on"]=bool(ui_item.checkBox_avgBlur.isChecked()),                
        Para_dict["Average blur Lower"]=int(ui_item.spinBox_avgBlurMin.value()),
        Para_dict["Average blur Higher"]=int(ui_item.spinBox_avgBlurMax.value()),
        Para_dict["Gauss blur on"]= bool(ui_item.checkBox_gaussBlur.isChecked()) ,                
        Para_dict["Gauss blur Lower"]=int(ui_item.spinBox_gaussBlurMin.value()),
        Para_dict["Gauss blur Higher"]=int(ui_item.spinBox_gaussBlurMax.value()),
    except Exception as e:
        message(e)
    try:
        Para_dict["Motion blur on"]=bool(ui_item.checkBox_motionBlur.isChecked()),
        motionBlur_kernel = str(ui_item.lineEdit_motionBlurKernel.text())
        motionBlur_angle = str(ui_item.lineEdit_motionBlurAngle.text())
        motionBlur_kernel = tuple(ast.literal_eval(motionBlur_kernel)) #translate string in the lineEdits to a tuple
        motionBlur_angle = tuple(ast.literal_eval(motionBlur_angle)) #translate string in the lineEdits to a tuple
        Para_dict["Motion blur Kernel"]=motionBlur_kernel,               
        Para_dict["Motion blur Angle"]=motionBlur_angle,          
    except Exception as e:
        message(e)

    Para_dict["Epoch_Started_Using_These_Settings"]=np.nan,
    try:
        Para_dict["expert_mode"]=bool(ui_item.groupBox_expertMode.isChecked()),
        Para_dict["batchSize_expert"]=int(ui_item.spinBox_batchSize.value()),
        Para_dict["epochs_expert"]=int(ui_item.spinBox_epochs.value()),
    except Exception as e:
        message(e)
    try:
        Para_dict["learning_rate_expert_on"]=bool(ui_item.groupBox_learningRate.isChecked()),
        Para_dict["learning_rate_const_on"]=bool(ui_item.radioButton_LrConst.isChecked()),
        Para_dict["learning_rate_const"]=float(ui_item.doubleSpinBox_learningRate.value()),
        Para_dict["learning_rate_cycLR_on"]=bool(ui_item.radioButton_LrCycl.isChecked()),
    except Exception as e:
        message(e)
    try:
        Para_dict["cycLrMin"]=float(ui_item.lineEdit_cycLrMin.text()),
        Para_dict["cycLrMax"]=float(ui_item.lineEdit_cycLrMax.text()),
    except:
        Para_dict["cycLrMin"]=np.nan,
        Para_dict["cycLrMax"]=np.nan,
    try:
        Para_dict["cycLrMethod"] = str(ui_item.comboBox_cycLrMethod.currentText()),
        Para_dict["clr_settings"] = ui_item.clr_settings,
    except Exception as e:
        message(e)
    try:
        Para_dict["learning_rate_expo_on"]=bool(ui_item.radioButton_LrExpo.isChecked()) ,
        Para_dict["expDecInitLr"]=float(ui_item.doubleSpinBox_expDecInitLr.value()),
        Para_dict["expDecSteps"]=int(ui_item.spinBox_expDecSteps.value()),
        Para_dict["expDecRate"]=float(ui_item.doubleSpinBox_expDecRate.value()),
    except Exception as e:
        message(e)
    try:
        Para_dict["loss_expert_on"]= bool(ui_item.checkBox_expt_loss.isChecked()),
        Para_dict["loss_expert"]=str(ui_item.comboBox_expt_loss.currentText()).lower(),
        Para_dict["optimizer_expert_on"]=bool(ui_item.checkBox_optimizer.isChecked()),
        Para_dict["optimizer_expert"]=str(ui_item.comboBox_optimizer.currentText()).lower(),                
        Para_dict["optimizer_settings"]=ui_item.optimizer_settings,                
    except Exception as e:
        message(e)
    try:
        Para_dict["paddingMode"]=str(ui_item.comboBox_paddingMode.currentText())#.lower(),                
    except Exception as e:
        message(e)
    try:
        Para_dict["train_last_layers"]=bool(ui_item.checkBox_trainLastNOnly.isChecked()),
        Para_dict["train_last_layers_n"]=int(ui_item.spinBox_trainLastNOnly.value())     ,
        Para_dict["train_dense_layers"]=bool(ui_item.checkBox_trainDenseOnly.isChecked()),
        Para_dict["dropout_expert_on"]=bool(ui_item.checkBox_dropout.isChecked()),
    except Exception as e:
        message(e)
    try:
        dropout_expert = str(ui_item.lineEdit_dropout.text()) #due to the validator, there are no squ.brackets
        dropout_expert = "["+dropout_expert+"]"
        dropout_expert = ast.literal_eval(dropout_expert)        
    except:
        dropout_expert = []
    try:
        Para_dict["dropout_expert"]=dropout_expert,
        Para_dict["lossW_expert_on"]=bool(ui_item.checkBox_lossW.isChecked()),
    except Exception as e:
        message(e)
    try:
        lossW_expert = str(ui_item.lineEdit_lossW.text())
        SelectedFiles = ui_item.items_clicked()
        class_weight = ui_item.get_class_weight(SelectedFiles,str(ui_item.lineEdit_lossW.text()),custom_check_classes=True)
        if type(class_weight)==list:
            #There has been a mismatch between the classes described in class_weight and the classes available in SelectedFiles!
            lossW_expert = class_weight[0] #overwrite 
            class_weight = class_weight[1]
            print("class_weight:" +str(class_weight))
            print("There has been a mismatch between the classes described in \
                  Loss weights and the classes available in the selected files! \
                  Hence, the Loss weights are set to Balanced")
        Para_dict["lossW_expert"]=lossW_expert,
        Para_dict["class_weight"]=class_weight,
    except Exception as e:
        message(e)
    try:
        Para_dict["metrics"]=ui_item.get_metrics(1),       
        if norm == "StdScaling using mean and std of all training data":                                
            #This needs to be saved into Para_dict since it will be required for inference
            Para_dict["Mean of training data used for scaling"]=np.nan,
            Para_dict["Std of training data used for scaling"]=np.nan,
    except Exception as e:
        message(e)

    return Para_dict

class Ui_Updates(QtWidgets.QWidget):
    def setupUi(self, Updates):
        Updates.setObjectName("Updates")
        Updates.resize(565, 514)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(os.path.join(dir_root,"art",Default_dict["Icon theme"],"main_icon_simple_04_update_01.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)

        self.gridLayout_5 = QtWidgets.QGridLayout(Updates)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.pushButton_iconUpdate = QtWidgets.QPushButton(Updates)
        self.pushButton_iconUpdate.setText("")

        self.pushButton_iconUpdate.setIcon(icon)
        self.pushButton_iconUpdate.setIconSize(QtCore.QSize(48, 48))
        self.pushButton_iconUpdate.setFlat(True)
        self.pushButton_iconUpdate.setObjectName("pushButton_iconUpdate")
        self.gridLayout_5.addWidget(self.pushButton_iconUpdate, 0, 0, 1, 1)
        self.groupBox_majorVersionInfo = QtWidgets.QGroupBox(Updates)
        self.groupBox_majorVersionInfo.setObjectName("groupBox_majorVersionInfo")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox_majorVersionInfo)
        self.gridLayout.setObjectName("gridLayout")
        self.label_yourVersion = QtWidgets.QLabel(self.groupBox_majorVersionInfo)
        self.label_yourVersion.setObjectName("label_yourVersion")
        self.gridLayout.addWidget(self.label_yourVersion, 0, 0, 1, 1)
        self.lineEdit_yourVersion = QtWidgets.QLineEdit(self.groupBox_majorVersionInfo)
        self.lineEdit_yourVersion.setEnabled(False)
        self.lineEdit_yourVersion.setObjectName("lineEdit_yourVersion")
        self.gridLayout.addWidget(self.lineEdit_yourVersion, 0, 1, 1, 1)
        self.textBrowser_majorVersionInfo = QtWidgets.QTextBrowser(self.groupBox_majorVersionInfo)
        self.textBrowser_majorVersionInfo.setEnabled(True)
        self.textBrowser_majorVersionInfo.setObjectName("textBrowser_majorVersionInfo")
        self.gridLayout.addWidget(self.textBrowser_majorVersionInfo, 1, 0, 1, 2)
        self.gridLayout_5.addWidget(self.groupBox_majorVersionInfo, 0, 1, 1, 1)
        self.groupBox_minorUpdates = QtWidgets.QGroupBox(Updates)
        self.groupBox_minorUpdates.setObjectName("groupBox_minorUpdates")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.groupBox_minorUpdates)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.groupBox_localDisk = QtWidgets.QGroupBox(self.groupBox_minorUpdates)
        self.groupBox_localDisk.setCheckable(False)
        self.groupBox_localDisk.setObjectName("groupBox_localDisk")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox_localDisk)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.comboBox_updatesOndevice = QtWidgets.QComboBox(self.groupBox_localDisk)
        self.comboBox_updatesOndevice.setObjectName("comboBox_updatesOndevice")
        self.gridLayout_2.addWidget(self.comboBox_updatesOndevice, 0, 0, 1, 1)
        self.pushButton_findFile = QtWidgets.QPushButton(self.groupBox_localDisk)
        self.pushButton_findFile.setObjectName("pushButton_findFile")
        self.gridLayout_2.addWidget(self.pushButton_findFile, 0, 1, 1, 1)
        self.pushButton_installOndevice = QtWidgets.QPushButton(self.groupBox_localDisk)
        self.pushButton_installOndevice.setObjectName("pushButton_installOndevice")
        self.gridLayout_2.addWidget(self.pushButton_installOndevice, 1, 0, 1, 2)
        self.gridLayout_4.addWidget(self.groupBox_localDisk, 1, 0, 1, 2)
        self.groupBox_online = QtWidgets.QGroupBox(self.groupBox_minorUpdates)
        self.groupBox_online.setCheckable(False)
        self.groupBox_online.setObjectName("groupBox_online")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox_online)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.comboBox_updatesOnline = QtWidgets.QComboBox(self.groupBox_online)
        self.comboBox_updatesOnline.setObjectName("comboBox_updatesOnline")
        self.gridLayout_3.addWidget(self.comboBox_updatesOnline, 0, 0, 1, 1)
        self.pushButton_installOnline = QtWidgets.QPushButton(self.groupBox_online)
        self.pushButton_installOnline.setObjectName("pushButton_installOnline")
        self.gridLayout_3.addWidget(self.pushButton_installOnline, 1, 0, 1, 1)
        self.gridLayout_4.addWidget(self.groupBox_online, 1, 2, 1, 1)
        self.gridLayout_5.addWidget(self.groupBox_minorUpdates, 1, 1, 1, 1)
        self.groupBox_minorUpdates.raise_()
        self.groupBox_majorVersionInfo.raise_()
        self.pushButton_iconUpdate.raise_()

        self.retranslateUi(Updates)
        QtCore.QMetaObject.connectSlotsByName(Updates)


        ###Icons###
        self.pushButton_installOndevice.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"install_fromLocal.png")))
        self.pushButton_installOnline.setIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"install_fromGitHub.png")))
        
    def retranslateUi(self, Updates):
        _translate = QtCore.QCoreApplication.translate
        Updates.setWindowTitle(_translate("Updates", "Update AIDeveloper",None))
        self.groupBox_majorVersionInfo.setTitle(_translate("Updates", "General information",None))
        self.label_yourVersion.setText(_translate("Updates", "Your version",None))
        self.groupBox_minorUpdates.setTitle(_translate("Updates", "Select update",None))
        self.groupBox_localDisk.setTitle(_translate("Updates", "From local disk",None))
        self.pushButton_findFile.setText(_translate("Updates", "Add file"))
        self.pushButton_installOndevice.setText(_translate("Updates", "Install",None))
        self.groupBox_online.setTitle(_translate("Updates", "From online repository",None))
        self.pushButton_installOnline.setText(_translate("Updates", "Download + Install",None))























#if __name__ == "__main__":
#    app = QtWidgets.QApplication(sys.argv)
#    Form = QtWidgets.QWidget()
#    ui = Fitting_Ui()
#    ui.setupUi(Form)
#    Form.show()
#    sys.exit(app.exec_())
