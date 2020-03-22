
![alt text](https://github.com/maikherbig/AIDeveloper/blob/master/art/main_icon_simple_04_text2.png "AIDeveloper Logo with Text")  

AIDeveloper is a software tool that allows you to train, evaluate and apply deep neural nets 
for image classification within a graphical user-interface (GUI).  

## Installation 
There is a tutorial video (ca. 1min. short) on YouTube.  
In this video, I show you how to get AIDeveloper running on your Windows 7 or Windows 10 PC.  
[![Alternate Text](/art/Youtube_Link_Tutorial0_v01.png)](https://youtu.be/uqygHsVlCCM "AIDeveloper Tutorial 0")  

  
**_If you dont want to watch the video:_**   
Go through the following 6 steps and you are good to go:    
* Go to https://github.com/maikherbig/AIDeveloper/releases
* Download a zip-file (this file contains the **_standalone executable_**)   
* Unzip it  
* Go into the unzipped folder and scroll down until you find an executable (full name is for example "AIDeveloper_0.0.6.exe")  
* DoubleClick this .exe to run it (no installation is required) 

## Basic usage
There is a tutorial video (ca. 13min. short) on YouTube.  
In this video only the very basic functionality of AID is presented. Furthermore, AIDeveloper contains many tooltips, which should help you to proceed further.  
[![Alternate Text](/art/Youtube_Link_Tutorial1_v04.png)](https://youtu.be/dvFiSRnwoto "AIDeveloper Tutorial 1")

  
  
In a second tutorial (28min), the 'Expert' options of AID are exploited to perform transfer learning. 
First, an existing CNN is loaded into AID. This CNN was trained previously on CIFAR-10 (grayscale) until a validation accuracy of 83%. Next, the Fashion-MNIST dataset is loaded into AID and training of the model is continued on this dataset. At the beginning, only the last layer of the CNN is trained, but later more and more layers are included into training. Also the dropout rates are optimized during the training process until a possibly record breaking testing accuracy of above 92% is reached. 
[![Alternate Text](art/Youtube_Link_Tutorial2_v04.png)](https://youtu.be/NWhv4PF0C4g "AIDeveloper Tutorial 2")


## Prerequisites

Since version 0.0.6, the standalone executables [executables](https://github.com/maikherbig/AIDeveloper/releases) of AIDeveloper are compatible to Windows, Mac and Linux.

The script based version was tested using Python 3.5 on Windows, Mac and Linux. See below to find installation instructions.

## Installation instructions to run AIDeveloper from script
**_you only need to do this if you are a developer/programmer_**
* Get a Python distribution. AIDeveloper was developed using Anaconda3 5.3.1 64bit. You can find this particular version in the installer archive of Anconda: https://repo.continuum.io/archive/
* Download the entire GitHub repository of AIDeveloper: find the green button on the top right ("Clone or download"), click it and then click "Download ZIP"
* Unzip the folder for example to C:\Users\MyPC\Downloads\AIDeveloper_0.0.6
* open Anaconda Prompt
* navigate to the folder where you put AIDeveloper `cd C:\Users\MyPC\Downloads\AIDeveloper_0.0.6`
* Generate an environment using the provided .yml file (AID_myenv_Win7.yml): `conda env create -f AID_myenv_Win7.yml`
* This will create a new environment called "myenv"
* Activate it: `conda activate myenv`
* Run AIDeveloper using `python AIDeveloper_0.0.6.py`


