
![alt text](https://github.com/maikherbig/AIDeveloper/blob/master/art/main_icon_simple_04_text2.png "AIDeveloper Logo with Text")  

AIDeveloper is a software tool that allows you to train, evaluate and apply deep neural nets 
for image classification within a graphical user-interface (GUI).  
Please find **_standalone executables_** of AIDeveloper here:  
https://github.com/maikherbig/AIDeveloper/releases  
After the following 4 steps you are good to go:  
* Download the zip-file   
* Unzip it  
* Search for the executable (.exe) (full name is for example "AIDeveloper_0.0.4.exe")  
* ---(Attention: if you are already using Keras: the file "home/.keras/keras.json" will be overwritten!!!)---  
* DoubleClick this .exe to run it (no installation is required)  

## Getting Started

There is a tutorial video (ca. 13min. short) which you either watch on YouTube:  
[![Alternate Text](https://github.com/maikherbig/AIDeveloper/blob/master/art/Youtube_Link_03.png)]({https://youtu.be/dvFiSRnwoto} "AIDeveloper Tutorial 1")


or you can download from this GitHub page:   
* Find the green button on the top right ("Clone or download"), click it and then click "Download ZIP"
* Unzip the folder and find the file "Tutorial_01_Basics.mp4"  

In this video only the very basic functionality of AID is presented. Furthermore, AIDeveloper contains many tooltips, which should help you to proceed. 

## Prerequisites

Currently, the standalone executable for AIDeveloper is only compatible with Windows 7 and Windows 10.  
The script based version was tested using Python 3.5 and Windows 7. See below to find installation instructions.

## Installation instructions to run AIDeveloper from script

* Get a python distribution. AIDeveloper was developed using Anaconda3 5.3.1 64bit, which you can download here:
https://www.anaconda.com/distribution/
* Download the entire Github repository of AIDeveloper: find the green button on the top right ("Clone or download"), click it and then click "Download ZIP"
* Unzip the folder for example to C:\Users\MyPC\Downloads\AIDeveloper_0.0.4
* open Anaconda Prompt
* navigate to the folder where you put AIDeveloper `cd C:\Users\MyPC\Downloads\AIDeveloper_0.0.4`

* Generate an environment using the provided .yml file (AID_myenv_Win7.yml): `conda env create -f AID_myenv_Win7.yml`
* This will create a new environment called "myenv"
* Activate it: `conda activate myenv`
* Run AIDeveloper using `python AIDeveloper_0.0.4.py`


