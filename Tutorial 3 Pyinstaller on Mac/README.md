![alt text](https://github.com/maikherbig/AIDeveloper/blob/master/art/Logos_div_v04.png "Keras TensorFlow OpenCV Scikit-learn Qt PyInstaller Logos")  

Here, I show you how you can 
* set up a Python development environment for AIDeveloper
* use PyInstaller to freeze AIDeveloper to a standalone executable.
The follwing instructions guide you step by step to create an environment that includes packages such as
* Keras  
* Tensorflow  
* PyQt5  
* Scikit-learn  
* Open-CV  
* SciPy  

If you never used your Mac for Python programming you may first want to update Python. For that you first need Homebrew.
* Open Terminal to install homebrew:  
`/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"`
* Use Homebrew to update Python:  
`brew reinstall python`  
* Update pip (a Python package that helps you to install other Python packages)  
`sudo easy_install pip`  
  
* Install pyenv  
`curl https://pyenv.run | bash`    

* Install XCode, by running the following command in your terminal and following the installation instructions  
`xcode-select --install`  
  
* Prepare for virtual environment:  
`export PYENV_ROOT="$HOME/.pyenv"`  
`export PATH="$PYENV_ROOT/bin:$PATH"`  
`eval "$(pyenv init -)"`  
`eval "$(pyenv virtualenv-init -)"`   
  
* Create a virtual environment with shared libraries:  
`env PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.5.9`  
* Activate that environment  
`pyenv local myenv2`  
* Install all required packages:
`pyenv virtualenv 3.5.9 myenv2`  
`pyenv local myenv2`  
`pip install --upgrade pip`  
`pip install --upgrade setuptools`    
`pip install numpy==1.15.2 scikit-learn==0.20.0`  
`pip install python-dateutil==2.7.5`  
`pip install keras==2.2.2`  
`pip install h5py==2.8.0`  
`pip install pandas==0.24.0 psutil==5.4.7`  
`pip install dclab==0.22.1`  
`pip install tensorflow==1.10.0`  
`pip install mkl==2018.0.3`  
`pip install Pillow==5.4.1`  
`pip install protobuf==3.6.0`  
`pip install pyqtgraph==0.10.0`  
`pip install pyqt5==5.9.2`  
`pip install openpyxl==2.5.6`  
`pip install xlrd==1.1.0`  
`pip install keras2onnx==1.4.0`  
`pip install numpy==1.15.2`  
`pip install setuptools==44.0.0`  
  
* To install PyInstaller, there are 3 options:  
* Option 1, install a particular version from GitHub:  
`pip install https://github.com/pyinstaller/pyinstaller/tarball/pyup/scheduled-update-2020-03-01`  
* Option 2 (shown in video), if this directory does not exist anymore, you can download this version of PyInstaller from my GitHub   repository (AIDeveloper/Tutorial 3 PyInstaller on Mac/pyinstaller-pyup-scheduled-update-2020-03-01.zip). Install it using:  
`pip install pyinstaller-pyup-scheduled-update-2020-03-01.zip`  
* Option 3: install the latest development version:  
`pip install https://github.com/pyinstaller/pyinstaller/tarball/develop`  
`pip install ffmpeg==1.4`  
`pip install libopencv`  
`pip install imageio==2.4.1`  
`pip install h5py==2.8.0`  
`pip install opencv-contrib-python-headless==4.1.1.26`  
`pip install tf2onnx==1.4.1`  
`pip install pyqt5==5.9.2`  
`pip install numpy==1.15.2`  
  
* Open a new Terminal at the folder containing all AIDeveloper files and the .spec file  
* To start PyInstaller, use the following command:  
`pyinstaller AIDeveloper_0.0.6_keep.spec`  
* (you can also find the file in Tutorial 3 PyInstaller on Mac/AIDeveloper_0.0.6_files.zip)  
 

When PyInstaller finished, there will be an object "AIDeveloper_0.0.6" with icon in "dist".  
Right click on the AIDeveloper_0.0.6 (the one with icon) and click "Show Package Contents".  
Find the folder called "MacOS" in /dist/AIDeveloper_0.0.6.app/Contents  
We need to copy some files into this "MacOS" folder:  
Copy following files:  
-AIDeveloper_Settings.json, -conda_list.txt, -Empty.rtdc, -layout_dark_notooltip.txt, -layout_dark.txt, -layout_darkorange_notooltip.txt, -layout_darkorange.txt, -main_icon_simple_04_256.icns, -model_zoo.py  
and the following folders:  
-\art, -\keras_metrics 
and paste them (files as well as folders) into "MacOS".  
We need two more folders to get Onnx working:  
- Either unzip the Onnx.zip folder, which you find in my GitHub repository: Tutorial 3 PyInstaller on Mac/Onnx.zip and copy both folders  
- or go into you home directoy and then to: /.pyenv/versions/3.5.9/envs/myenv2/lib/python3.5/site-packages  
and find the onnx and onnx-1.6.0.dist-info folder in there. Copy them and paste both folders into the MacOS folder. There will be a warning because of onnx: select "Merge".  
  
DONE! You can now double-click the AIDeveloper_0.0.6 with icon and it will run! Alternatively you can also start it from console by right-clicking on "MacOS" and selecting "New Terminal at Folder" and then typing in the console    
`./AIDeveloper_0.0.6` #(without .py at the end)
