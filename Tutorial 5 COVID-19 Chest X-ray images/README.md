# Tutorial 5: Diagnosis of COVID-19 using chest X-ray images

## Inspect the COVID-19 dataset  
In this tutorial, I want to show you how you can build a classification algorithm that detects illnesses based on chest X-ray images.  
On [this](https://github.com/ieee8023/covid-chestxray-dataset) website, I found chest X-ray images of COVID-19 patients.
Currently, the dataset is very small (<160 images) and unbalanced (only 1 image for "healthy" but >80 for 'COVID-19').
At the same time, the classification problem is quite complex.
Therefore, modern regularization techniques need to be employed to prevent overfitting.

As this is an image dataset, one can artificially increase the size of the dataset by mathematically altering the images
slightly such as random rotations and brightness or contrast changes. This method is called image augmentation.
Furthermore, one can employ a method termed transfer learning.
The idea behind transfer learning is to pre-train a CNN using a different (but ideally similar) dataset.
  
## Load a similar dataset from Kaggle and train a model  
To do so, I first downloaded a dataset from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).
This dataset contains chest X-ray images of "healthy" and "pneumonia" patients.
To equalize the dimension of these images, [I prepared a python script that you can find in this repository](https://github.com/maikherbig/AIDeveloper/blob/master/Tutorial%205%20COVID-19%20Chest%20X-ray%20images/01_DataPrep-Kaggle.py).

After loading the data into AIDeveloper, I tuned the image augmentation parameters to sensible values
by frequently checking example-images. All parameters are tracked in a meta-file, which 
I uploaded to [fighshare](https://doi.org/10.6084/m9.figshare.12040599) -> 01_Kaggle/M01_LeNet5do_300pix_2cl_meta.xlsx.  

Alternatively, you can just jump to the corresponding sequence in the video to
find out which values I used. I run AIDeveloper from script on a machine with GPU and CUDA installed.
This allowed me to train for 488 epochs in only 1.5h.  
A model at a very early epoch already resulted in a high accuracy. Since models
tend to become less applicable to different datasets, the longer they are trained,
I decided to apply this model to another similar dataset.  

## Check robustness of model by testing against CheXpert dataset  
After searching for another similar dataset, I found the [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) dataset.
CheXpert contains X-ray images of patients with different diseases, including 'pneumonia' and also images of "healthy" individuals.
Since "healthy" and "pneumonia" were also contained in the Kaggle dataset, the CheXpert dataset is the ideal source to
test the  model which was trained using the Kaggle dataset.
To download CheXpert one has to sign up at their [webpage](https://stanfordmlgroup.github.io/competitions/chexpert/) and you
will receive an email with download links. I downloaded the downsampled version of the dataset (11GB).
Again, the images are not of equal dimension. Therefore, I wrote a Python script to
prepare the dataset, which you can find in my [repository](https://github.com/maikherbig/AIDeveloper/blob/master/Tutorial%205%20COVID-19%20Chest%20X-ray%20images/02_DataPrep-CheXpert.py).  
    
After loading the contents of the "train"-folder into AIDeveloper, I used the "Assess Model" - Tab
to load the model "M01_LeNet5do_300pix_2cl_33.model" and hit the "Update plots" button
to apply the model to the CheXpert dataset.  
The resulting accuracy is only 0.6, showing that the model does not perform well on this new dataset. In other words the model is not
robust for the differences between the datasets.
Therefore I would assume that this model would not be the ideal starting point to perform transfer learning.
  
## Train model on the larger CheXpert dataset  
Since the CheXpert dataset is larger than the dataset from Kaggle, one could
expect to obtain a more robust model when training on CheXpert.  
To do so, first, the training data from CheXpert is loaded into AIDeveloper.
Next, the data from the "test"-folder from the Kaggle dataset is loaded into AIDeveloper and selected to be the validation data.

The advantage when training on CheXpert and validating using the Kaggle dataset is that we obtain after each training iteration ("epoch")
the validation accuracy, which tells immediately, if the model is robust for the differences between the CheXpert and Kaggle dataset.

To allow for reproducibility, I uploaded the meta-file, which contains all 
parameters and settings I used in AIDeveloper
to [fighshare](https://doi.org/10.6084/m9.figshare.12040599) -> 02_Train-CheX_Valid-Kaggle/M02_LeNet5do_300pix_2cl_meta.xlsx.
Furthermore, I uploaded the three best models.

## Apply CheXpert model to the COVID-19 dataset
Before we can test these models on the COVID19 dataset, it is necessary to prepare the
dataset identically as the other dataset before. For this purpose, I prepared
a Pyton script, which you can find in this [repository](https://github.com/maikherbig/AIDeveloper/blob/master/Tutorial%205%20COVID-19%20Chest%20X-ray%20images/03_DataPrep-COVID.py).
Next, I moved two images of "SARS" and 15 images of "COVID-19" to a separate
validation set. After loading the training set into AIDeveloper, one can use the
"Assess Model" to load the best model (index 520), which was trained using the CheXpert dataset. You can 
download this model from [fighshare](https://doi.org/10.6084/m9.figshare.12040599) -> 02_Train-CheX_Valid-Kaggle/M02_LeNet5do_300pix_2cl_520.model. 

Surprisingly, the model classifies the only available image of the class "healthy" correctly.
Furthermore, 67% (57 of 85) images of COVID-19 patients are predicted into the class "pneumonia".
Maybe, there are similarities between pneumonia and COVID-19, resulting in this classification behaviour.

## Optimize model for COVID-19
So far, the model has never been trained on images of COVID-19 patients, but only on images of "healthy" and "pneumonia" patients.
Now, I would optimize the model only using data from the COVID-19 [dataset](https://github.com/ieee8023/covid-chestxray-dataset).
This dataset contains images of multiple diagnoses, but mostly of COVID-19. This
means all other classes beside COVID-19 contains only very few (sometimes a single) images.
In order to create a class with a sufficient amount of images, I loaded all the images
of ARDS, SARS, and Streptococcus into AIDeveloper and assigned all to the same class ("0").
The images of COVID-19 were assigned to class 1.
This configuration will allow to train a model that can distinguish COVID-19 from ARDS, SARS and Streptococcus.
    
Now, the "Load and continue" option in AIDeveloper is used to load the existing model "M02_LeNet5do_300pix_2cl_520.model".
On the "Expert"-Tab, I set the dropout rates to 0.8 in order to prevent overfitting. Furthermore,
I froze all but the last two layers of the network. This means that during training, only these last
two layers will be optimized. The intuition behind this method is that during the prior training
using CheXpert, we already optimized the convolutional layers of the network. As we are now using
this model for a very similar task, we can expect that these convolutional layers still compute useful
output. Therefore, these layers don't need to be optimized further, resulting in 
less free parameters and therefore less risk of overfitting. 
  
After training for a couple of epochs, models with a validation accuracy of 0.7 were reached, followed
by a plateau, indicating that the model does not learn anymore. 
Therefore, I reduced the dropout rates. Reducing dropout rates can be done in 
AIDeveloper on the "Expert"-Tab even during the training process. After reducing the 
dropout rates again (to 0.65), the validation accuracy dropped, indicating overfitting.
Therefore, I terminated the fitting process and continued to assess the best models
in more detail.
The meta-file with all parameters and settings as well as the best model are available 
on [fighshare](https://doi.org/10.6084/m9.figshare.12040599) -> 03_COVID-Model.
I loaded the best model ("M03_LeNet5do_Pretr1_300pix_2cl_41.model") using
the "Asses Model"-tab and used the button "Update Plots" to apply the model to 
the validation set and to obtain the confusion matrix.
From the confusion matrix I learned that 10 of 15 COVID-19 cases are predicted correctly.

Due to the small size of the COVID-19 dataset, the performance and robustness
of this model cannot be tested thoroughly. 
    
I hope this tutorial inspires people to try AIDeveloper to 
solve biomedical problems 










