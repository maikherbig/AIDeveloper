
![alt text](art/main_icon_simple_04_text2.png "AIDeveloper Logo with Text")  

AIDeveloper is a software tool that allows you to train, evaluate and apply deep neural nets 
for image classification within a graphical user-interface (GUI).  

# Overview  
![Alt Text](art/Intro-v07.gif)
# Installation  
There is a tutorial video (44 seconds short) on YouTube.  
In this video, I show you how to get AIDeveloper running on your PC.  
[![Alternate Text](/art/Youtube_Link_Tutorial0_v01.png)](https://youtu.be/s5Kby9UuzL4 "AIDeveloper Tutorial 0")  

  
**_If you dont want to watch the video:_**   
Go through the following 5 steps and you are good to go:    
* Go to https://github.com/maikherbig/AIDeveloper/releases
* Download a zip-file (this file contains the **_standalone executable_**)   
* Unzip it  
* Go into the unzipped folder and scroll down until you find an executable (full name is for example "AIDeveloper_0.0.6.exe")  
* DoubleClick this .exe to run it (no installation is required) 

# Tutorials  
## Basic usage  
There is a tutorial video (ca. 13min. short) on YouTube.  
In this video only the very basic functionality of AID is presented. Furthermore, AIDeveloper contains many tooltips, which should help you to proceed further.  
[![Alternate Text](art/Youtube_Link_Tutorial1_v04.png)](https://youtu.be/dvFiSRnwoto "AIDeveloper Tutorial 1")
  
  
## Transfer learning  
In a second tutorial (28min), the 'Expert' options of AID are exploited to perform transfer learning.  
First, an existing CNN is loaded into AID. This CNN was trained previously on CIFAR-10 (grayscale) until a validation accuracy of 83%. Next, the Fashion-MNIST dataset is loaded into AID and training of the model is continued on this dataset. At the beginning, only the last layer of the CNN is trained, but later more and more layers are included into training. Also the dropout rates are optimized during the training process until a possibly record breaking testing accuracy of above 92% is reached.  
[![Alternate Text](art/Youtube_Link_Tutorial2_v04.png)](https://youtu.be/NWhv4PF0C4g "AIDeveloper Tutorial 2")
  
  
## Learning rate screening and learning rate schedules  
The learning rate (LR) is an important parameter when it comes to training neural networks. AID features a LR screening method which was originally suggested in
[this paper](https://arxiv.org/abs/1506.01186). That method allows you to find a LR that is suited well for your setting. Furthermore, this tutorial introduces LR schedules, which allow you to automatically adjust the learning rate during the training process. Besides exponentially decreasing learning rates, AID also features cyclical learning rates, which were also introduced in the [paper mentioned earlier](https://arxiv.org/abs/1506.01186).     
[![Alternate Text](art/Youtube_Link_Tutorial_LR_schedules.png)](https://youtu.be/cQSFFURAtPc "AIDeveloper Learning rate schedules and learning rate screening")  
  
  
## Detecting COVID-19 using chest X-ray images  
In this tutorial, AID is used to tackle a biomedical question that is currently of high interest: diagnosis of COVID-19. One problem is the scarcity of COVID-19 X-ray images, which results in a need of modern regularization techniques to prevent overfitting. First, two other large datasets are used to pre-train a model. Next, this model is optimized for images of COVID-19.
More information and step by step instructions are available [here](https://github.com/maikherbig/AIDeveloper/tree/master/Tutorial%205%20COVID-19%20Chest%20X-ray%20images).  
Furthermore, there is a video showing the analysis procedure from beginning to end:  
[![Alternate Text](art/Youtube_Link_Tutorial5_v03.png)](https://www.youtube.com/watch?v=KRDJBJD7CsA "AIDeveloper Tutorial 5")
  
  
## More tutorials  
[Adding models to the model zoo](https://www.youtube.com/watch?v=XboH-YsG6LA&t)  
[Create standalone using PyInstaller](https://figshare.com/articles/Krater_et_al_2020_Data_zip/9902636)  
[AIDeveloper on AWS with GPU support](https://www.youtube.com/watch?v=C3pMNAg68XQ&t)  
[Deploy a model using OpenCV](https://github.com/maikherbig/AIDeveloper/tree/master/Tutorial%20Deploy%20to%20OpenCV%20dnn)  

# Prerequisites  
Since version 0.0.6, standalone [executables](https://github.com/maikherbig/AIDeveloper/releases) of AIDeveloper are available. For Windows, a GPU-version is available that can detect and use NVIDIA GPUs (installation of CUDA is not required).

The script based version was tested using Python 3.9.10 on Windows and Mac. See below to find installation instructions.

# Installation instructions to run AIDeveloper from script
**_you only need to do this if you are a developer/programmer_**
* On Windows, install [Visual studio build tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/). During installation, make sure to check C++ Buildtools:
![alt text](https://github.com/maikherbig/AIDeveloper/blob/master/art/VS_Build_Tools.png "Installation of VS Build tools")
* Get a Python distribution. AIDeveloper was developed using Miniconda3 ("Miniconda3-py39_4.9.2-Windows-x86_64"). You can find this particular version in the installer archive of Miniconda: https://repo.anaconda.com/miniconda/
* Download the entire GitHub repository of AIDeveloper: find the green button on the top right ("Clone or download"), click it and then click "Download ZIP"
* Unzip the folder for example to C:\Users\MyPC\Downloads\AIDeveloper_0.3.0
* open Anaconda Prompt
### Option 1: Install dependencies using .yml file
* navigate to the folder where you put AIDeveloper `cd C:\Users\MyPC\Downloads\AIDeveloper_0.3.0`
* Generate an environment using the provided .yml file (AID_env_gpu_Win10.yml): `conda env create -f AID_env_gpu_Win10.yml`
* This will create a new environment called "aid3_spyder"
* Activate it: `conda activate aid_env_cpu`
* Run AIDeveloper using `python AIDeveloper.py`  
### Option 2: Manually install all dependencies:
```
conda create -n aid3_spyder python==3.9.10 spyder==5.2.1 typing-extensions==3.7.4.3
conda activate aid3_spyder
pip install --upgrade setuptools
pip install tensorflow-gpu==2.7.1
pip install scikit-learn==1.0.2
pip install dclab==0.39.9
pip install Pillow==9.0.0
pip install pandas==1.1.5
pip install psutil==5.9.0
pip install mkl==2022.0.2
#pip install pyqt5==5.12.3
pip install pyqtgraph==0.12.3
pip install imageio==2.13.5
pip install opencv-contrib-python-headless==4.5.5.62
pip install openpyxl==3.0.9
pip install xlrd==2.0.1
pip install keras2onnx==1.7.0
pip install libopencv==0.0.1
pip install ffmpeg==1.4
pip install tf2onnx==1.9.3
pip install Keras-Applications==1.0.8
```
# AIDeveloper in scientific literature  
[1]	[M. Kräter et al., “AIDeveloper: Deep Learning Image Classification in Life Science and Beyond,” Advanced Science, Mar. 2021.](https://onlinelibrary.wiley.com/doi/10.1002/advs.202003743)  
[2]	[A. A. Nawaz et al., “Intelligent image-based deformation-assisted cell sorting with molecular specificity,” Nat. Methods, May 2020.](https://rdcu.be/b4ow4)    
[3]	[T. Krüger et al., “Reliable isolation of human mesenchymal stromal cells from bone marrow biopsy specimens in patients after allogeneic hematopoietic cell transplantation,” Cytotherapy, vol. 22, no. 1, pp. 21–26, Jan. 2020.](https://www.ncbi.nlm.nih.gov/pubmed/31883948)  
[4]	[P. Oura et al., “Deep learning in forensic gunshot wound interpretation—a proof-of-concept study,” International Journal of Legal Medicine, vol. 135, pp. 2101–2106, Apr. 2021.](https://doi.org/10.1007/s00414-021-02566-3)  
[5]	[A. Walther et al., “Depressive disorders are associated with increased peripheral blood cell deformability: A cross-sectional case-control study (Mood-Morph),” medRxiv, Jul. 2021.](https://doi.org/10.1101/2021.07.01.21259846)  
# Citing AIDeveloper  
If you use AIDeveloper in a scientific publication, citation of the following paper is appreciated:  
[M. Kräter et al., “AIDeveloper: Deep Learning Image Classification in Life Science and Beyond,” Advanced Science, Mar. 2021.](https://onlinelibrary.wiley.com/doi/10.1002/advs.202003743)  
