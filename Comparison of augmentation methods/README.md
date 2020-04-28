# Comparison of the speed of multiple image augmentation functions

![alt text](https://github.com/maikherbig/AIDeveloper/blob/master/Comparison%20of%20augmentation%20methods/art/01_Quokka_Horiz_Vertical.png "01_Quokka")  

Image augmetation is a powerful technique, allowing you to increase the size  
of your dataset. Image augmentation is based on a mathematical alteration of 
the original images in a meaningful manner. For example, lets have a look at 
this quokka. Flipping the image along the horizontal axis is a useful 
opertation, as this image shows a scene that could appaer in the real world and 
would help durig training to obtain a more robust model. On contrary, flipping 
along the vertical axis results in images that one would probably never see in 
a real-world scenario.  

Datasets for deep learning purposes are typically on the size of thousands to  
millions of images, resulting in a high computational demand to perform such 
image augmentation operations.  

Here, I want to compare the speed of the following image augmentation algorithms:  
* [Keras ImageDataGenerator](https://keras.io/preprocessing/image/#imagedatagenerator-class)   
* [imgaug](https://imgaug.readthedocs.io/en/latest/)  
* [AIDevelopers aid_img.py](https://github.com/maikherbig/AIDeveloper/blob/master/AIDeveloper/aid_img.py)  

First, lets load the quokka image provided by the imgaug library:

<pre><code>
path = os.path.join("imgaug","quokka.jpg")
img = load_img(path,color_mode=color_mode.lower()) #This uses PIL and supports many many formats!
img = np.array(img)
images = []
for i in range(250):
    images.append(img)
images = np.array((images), dtype="uint8")
</code></pre>

