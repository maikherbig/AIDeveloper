This tutorial shows how to convert .cif files from ImageStream to .rtdc files and
how to use AIDeveloper and YouLabel to create a dataset and train a CNN model.
There are six steps; the five steps are explained in a video:


1. Install Java and Python environment for converting cif files (see [below](#install-java-and-python-environment))
2. Modify Python script to optimize conversion to .rtdc (see video)
3. Annotate data using [YouLabel](https://github.com/maikherbig/YouLabel/releases) (see video)
4. Use AIDeveloper to train CNN using Multi-Channel information (see video)
5. Use AIDeveloper to apply model to new data (see video)
6. Filter the data using probabilities, returned by the model (see video)


# Install Java and Python environment   
First, install Java development kit (JDK). Windows installers can be obtained [on this website](https://www.oracle.com/java/technologies/downloads/#jdk18-windows), or via this [download link](https://download.oracle.com/java/18/latest/jdk-18_windows-x64_bin.exe).  
Next, a Python distribution is required. I like to work with [miniconda](https://docs.conda.io/en/latest/miniconda.html). 
Using conda, create a Python environment using following commands:
```
conda create -n imagestream1 python==3.9 spyder==5 typing-extensions==3.7.4.3
conda activate imagestream1
pip install javabridge==1.0.19
pip install python-bioformats==4.0.5
pip install pandas==1.1.5
pip install opencv-contrib-python-headless==4.5.5.62
pip install openpyxl==3.0.9
pip install xlrd==2.0.1
pip install matplotlib==3.5.2
pip install joblib==1.1.0
pip install tqdm==4.64.0
pip install h5py==3.7.0  
```
