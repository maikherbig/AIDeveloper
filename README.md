# AIDeveloper

AIDeveloper is a software tool that allows you to train, evaluate and apply deep neural nets for image classification.
\nPlease find standalone executables of AIDeveloper here: https://github.com/maikherbig/AIDeveloper/releases
\nAfter the following 4 steps you are good to go:
\n-Download the zip-file 
\n-Unzip it
\n-Search for the executable (.exe) (full name is for example "AIDeveloper_0.0.4.exe")
\n--(Attention: if you are already using Keras: the file "home/.keras/keras.json" will be overwritten!!!)---
\n-DoubleClick this .exe to run it (no installation is required)

## Getting Started

\nThere is a tutorial video (ca. 13min. short) which you can download:
\n-Find the green button on the top right ("Clone or download"), click it and then click "Download ZIP"
\n-Unzip the folder and find the file "Tutorial_01_Basics.mp4"
\nIn this video only the very basic functionality of AID is presented.
\nFurthermore, AIDeveloper contains many tooltips, which hopefully help you. 

### Prerequisites

\nCurrently, the standalone executable for AIDeveloper is only available for Windows 7 and Windows 10.
\nThe script based version was tested using Python 3.5 on Windows 7. See below to find installation instructions.

### Installation instructions to run AIDeveloper from script
\nAIDeveloper was developed using Anaconda3 5.3.1 64bit, which you can download here:
https://www.anaconda.com/distribution/

\nAlso download this entire Github repository:
\n-Find the green button on the top right ("Clone or download"), click it and then click "Download ZIP"
\n-Unzip the folder for example to C:\Users\MyPC\Downloads\AIDeveloper_0.0.4
\n-open Anaconda Prompt
\n-navigate to this folder: "cd C:\Users\MyPC\Downloads\AIDeveloper_0.0.4"
\n-Generate an environment using the provided .yml file (AID_myenv_Win7.yml): "conda env create -f AID_myenv_Win7.yml"
\n-This will create a new environment called "myenv"
\n-Activate it: "conda activate myenv"
\n-Run AIDeveloper using "python AIDeveloper_0.0.4.py"




