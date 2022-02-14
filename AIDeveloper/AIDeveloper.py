# -*- coding: utf-8 -*-
"""
AIDeveloper
---------
@author: maikherbig
"""
import os,sys,gc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'#suppress warnings/info from tensorflow

if not sys.platform.startswith("win"):
    from multiprocessing import freeze_support
    freeze_support()
# Make sure to get the right icon file on win,linux and mac
if sys.platform=="darwin":
    icon_suff = ".icns"
else:
    icon_suff = ".ico"

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
from pyqtgraph import Qt

import aid_start
dir_root = os.path.dirname(aid_start.__file__)#ask the module for its origin
dir_settings = os.path.join(dir_root,"aid_settings.json")#dir to settings
Default_dict = aid_start.get_default_dict(dir_settings) 

try:
    splashapp = QtWidgets.QApplication(sys.argv)
    #splashapp.setWindowIcon(QtGui.QIcon("."+os.sep+"art"+os.sep+Default_dict["Icon theme"]+os.sep+"main_icon_simple_04_256.ico"))
    # Create and display the splash screen
    splash_pix = os.path.join(dir_root,"art",Default_dict["Icon theme"],"main_icon_simple_04_256"+icon_suff)
    splash_pix = QtGui.QPixmap(splash_pix)
    #splash_pix = QtGui.QPixmap("."+os.sep+"art"+os.sep+Default_dict["Icon theme"]+os.sep+"main_icon_simple_04_256"+icon_suff)
    splash = QtWidgets.QSplashScreen(splash_pix, QtCore.Qt.WindowStaysOnTopHint)
    splash.setMask(splash_pix.mask())
    splash.show()
except:
    pass

import aid_backbone

def main():
    #global app
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon(os.path.join(dir_root,"art",Default_dict["Icon theme"],"main_icon_simple_04_256"+icon_suff)))

    if Default_dict["Layout"] == "Dark":
        dir_layout = os.path.join(dir_root,"layout_dark.txt")#dir to settings
        f = open(dir_layout, "r") #I obtained the layout file from: https://github.com/ColinDuquesnoy/QDarkStyleSheet/blob/master/qdarkstyle/style.qss
        f = f.read()
        app.setStyleSheet(f)
    elif Default_dict["Layout"] == "DarkOrange":
        dir_layout = os.path.join(dir_root,"layout_darkorange.txt")#dir to settings
        f = open(dir_layout, "r") #I obtained the layout file from: https://github.com/nphase/qt-ping-grapher/blob/master/resources/darkorange.stylesheet
        f = f.read()
        app.setStyleSheet(f)
    else:
        app.setStyleSheet("")
    
    ui = aid_backbone.MainWindow()
    ui.add_app(app)
    ui.show()
    try:
        splash.finish(ui)
    except:
        pass

    ret = app.exec_()
    sys.exit(ret)

if __name__ == '__main__':
    main()
