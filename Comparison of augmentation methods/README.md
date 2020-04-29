# Comparison of the speed of multiple image augmentation functions

![alt text](https://github.com/maikherbig/AIDeveloper/blob/master/Comparison%20of%20augmentation%20methods/art/01_Quokka_Horiz_Vertical.png "01_Quokka")  

Image augmetation is a powerful technique, allowing to artifically increase the size
of a dataset. Image augmentation is based on a mathematical alteration of
the original images, which should ideally be done in a meaningful manner. 
For example, for the image of the quokka (shown above), horitontal flipping
is a useful operation, since the resulting image shows a scene that could 
appaer in the real world. Such an augmentation operation would would help durig training 
to obtain a more robust model. On contrary, flipping along the vertical axis 
results in an image that one would probably never see in a real-world scenario.  

Fortunately,  there are multiple libraries readily availabe to perform
image augmentation. In the inital release of AIDeveloper (v.0.0.4), the Keras
ImageDataGenerator was used, but interestingly, the computational time for
augmenting images was often simiar to the time required to train the neural
net. Considering the continuously growing amount of available data, more efficient
algorithms are needed.  

Here, I want to compare the speed of the following image augmentation algorithms:  
* [Keras ImageDataGenerator](https://keras.io/preprocessing/image/#imagedatagenerator-class)   
* [imgaug](https://imgaug.readthedocs.io/en/latest/)  
* [AIDevelopers aid_img.py](https://github.com/maikherbig/AIDeveloper/blob/master/AIDeveloper/aid_img.py)  
  

First load the quokka image provided by the imgaug library:

<pre><code>color_mode = "RGB"
path = os.path.join("imgaug","quokka.jpg")
img = load_img(path,color_mode=color_mode.lower()) #This uses PIL and supports many many formats!
img = np.array(img)
images = []
for i in range(250): # Replicate the image 250 times
    images.append(img)
images = np.array((images), dtype="uint8")</code></pre>

## Affine augmentation  
Define some parameters and the imgaug augmentation function:  
<pre><code>v_flip = True # bool, if random vertical flipping should be applied
h_flip = True # bool, if random horizontal flipping should be applied
rotation = 45 #degrees of random rotation
width_shift = 0.2 # shift the image left right
height_shift = 0.2 # shift the image up down
zoom = 0.2 # random zooming in range
shear = 0.2 # random shear in range

#For imgaug, define a function that performs affine augmentations in sequence  
def imgaug_affine(images,v_flip,h_flip,rotation,width_shift,height_shift,zoom,shear,backend):
    v_flip_imgaug = 0.5 if v_flip==True else 0.0
    h_flip_imgaug = 0.5 if h_flip==True else 0.0

    #Imgaug image augmentation pipeline for affine augmentation
    gen = imgaug.augmenters.Sequential([
            imgaug.augmenters.Fliplr(h_flip_imgaug), #flip 50% of the images horizontally
            imgaug.augmenters.Flipud(v_flip_imgaug), #flip 50% of the images vertically
            imgaug.augmenters.Affine(
                    rotate=(-rotation, rotation),
                    translate_percent={"x": (-width_shift, width_shift), "y": (-height_shift, height_shift)},
                    scale={"x": (1-zoom, 1+zoom), "y": (1-zoom, 1+zoom)},
                    shear=(-shear, shear),backend=backend)])
    return gen(images=images) #Imgaug image augmentation</code></pre>

### Affine augmentation: Keras ImageDataGenerator  
<pre><code>t1 = time.time()
gen = ImageDataGenerator(rotation_range=rotation,vertical_flip=v_flip,horizontal_flip=h_flip,width_shift_range=width_shift,height_shift_range=height_shift,zoom_range=zoom,shear_range=shear,dtype="unit8")
gen_keras = gen.flow(images, np.repeat(0,images.shape[0]), batch_size=images.shape[0])
images_keras = next(gen_keras)[0].astype(np.uint8)
t2 = time.time()
dt = t2-t1
print("Keras ImageDataGenerator "+str(np.round(dt,2))+"s")

fig=plt.figure(1)
for i in range(1,5):
    ax=fig.add_subplot(2,2,i)        
    ax.imshow(images_keras[i])
    ax.axis("off")
fig.suptitle("Keras ImageDataGenerator "+str(np.round(dt,2))+"s")
plt.savefig("02_Affine_Keras.png")
plt.close(1)</code></pre>

![alt text](https://github.com/maikherbig/AIDeveloper/blob/master/Comparison%20of%20augmentation%20methods/art/02_Affine_Keras.png "02_Affine augmentation_Keras_ImageDataGenerator")  

### Affine augmentation: imgaug using skimage backend  
<pre><code>t1 = time.time()
images_imgaug_sk = imgaug_affine(images,v_flip,h_flip,rotation,width_shift,height_shift,zoom,shear,backend="skimage")
t2 = time.time()
dt = t2-t1
print("imgaug (backend skimage) "+str(np.round(dt,2))+"s")

fig=plt.figure(1)
for i in range(1,5):
    ax=fig.add_subplot(2,2,i)        
    ax.imshow(images_imgaug_sk[i])
    ax.axis("off")
fig.suptitle("imgaug (backend skimage) "+str(np.round(dt,2))+"s")
plt.savefig("02_Affine_imgaug_sk.png")
plt.close(1)</code></pre>

![alt text](https://github.com/maikherbig/AIDeveloper/blob/master/Comparison%20of%20augmentation%20methods/art/02_Affine_imgaug_sk.png "02_Affine augmentation_Keras_ImageDataGenerator")  

### Affine augmentation: imgaug using OpenCV (cv2) backend   
<pre><code>t1 = time.time()
images_imgaug_cv = imgaug_affine(images,v_flip,h_flip,rotation,width_shift,height_shift,zoom,shear,backend="cv2")
t2 = time.time()
dt = t2-t1
print("imgaug (backend cv2) "+str(np.round(dt,2))+"s")

fig=plt.figure(1)
for i in range(1,5):
    ax=fig.add_subplot(2,2,i)        
    ax.imshow(images_imgaug_cv[i])
    ax.axis("off")
fig.suptitle("imgaug (backend cv2) "+str(np.round(dt,2))+"s")
plt.savefig("02_Affine_imgaug_cv2.png")
plt.close(1)</code></pre>

![alt text](https://github.com/maikherbig/AIDeveloper/blob/master/Comparison%20of%20augmentation%20methods/art/02_Affine_imgaug_cv2.png "02_Affine augmentation_Keras_ImageDataGenerator")  

### Affine augmentation: AIDeveloper's aid_img.py  
<pre><code>t1 = time.time()
images_aid = aid_img.affine_augm(images,v_flip,h_flip,rotation,width_shift,height_shift,zoom,shear) #Affine image augmentation
t2 = time.time()
dt = t2-t1
print("AIDeveloper "+str(np.round(dt,2))+"s")

fig=plt.figure(1)
for i in range(1,5):
    ax=fig.add_subplot(2,2,i)        
    ax.imshow(images_aid[i])
    ax.axis("off")
fig.suptitle("AIDeveloper "+str(np.round(dt,2))+"s")
plt.savefig("02_Affine_aid_img.png")
plt.close(1)</code></pre>
  
![alt text](https://github.com/maikherbig/AIDeveloper/blob/master/Comparison%20of%20augmentation%20methods/art/02_Affine_aid_img.png "02_Affine augmentation_Keras_ImageDataGenerator")  

## Brightness augmentation + noise  
For me it was actually rather surprising, that a simple linear transformation  
of the brightness of the image and (especially) adding random noise can take  
quite a while.  
Define some parameters and the imgaug augmentation function:  
<pre><code>add_lower = -50
add_upper = 50
mult_lower = 0.8
mult_upper = 1.2
gaussnoise_mean = 0
gaussnoise_scale = 3

def imgaug_brightness_noise(images,add_low,add_high,mult_low,mult_high,noise_mean,noise_std):
    seq = imgaug.augmenters.Sequential([
        imgaug.augmenters.Add((add_low, add_high)),
        imgaug.augmenters.Multiply((mult_low, mult_high)),
        imgaug.augmenters.AdditiveGaussianNoise(loc=noise_mean, scale=(noise_std, noise_std)),
    ])
    images = seq(images=images)
    return images</code></pre>

### Brightness augmentation: Keras ImageDataGenerator  
Unfortunately, there is no option in ImageDataGenerator to add random noise.  
Therefore I'll now only alter the brightness (only in case of ImageDataGenerator)

<pre><code>t1 = time.time()
gen = ImageDataGenerator(brightness_range=(mult_lower,mult_upper),dtype="unit8")
gen_keras = gen.flow(images, np.repeat(0,images.shape[0]), batch_size=images.shape[0])
images_keras = next(gen_keras)[0].astype(np.uint8)
t2 = time.time()
dt = t2-t1
print("Keras ImageDataGenerator "+str(np.round(dt,2))+"s")

fig=plt.figure(1)
for i in range(1,5):
    ax=fig.add_subplot(2,2,i)        
    ax.imshow(images_keras[i])
    ax.axis("off")
fig.suptitle("Keras ImageDataGenerator "+str(np.round(dt,2))+"s")
plt.savefig("02_Brightness_Keras.png")
plt.close(1)</code></pre>

![alt text](https://github.com/maikherbig/AIDeveloper/blob/master/Comparison%20of%20augmentation%20methods/art/02_Brightness_Keras.png "02_Affine augmentation_Keras_ImageDataGenerator")  

### Brightness augmentation: imgaug  
<pre><code>t1 = time.time()
images_imgaug = imgaug_brightness_noise(images,add_lower,add_upper,mult_lower,mult_upper,gaussnoise_mean,gaussnoise_scale)
t2 = time.time()
dt = t2-t1
print("imgaug "+str(np.round(dt,2))+"s")

fig=plt.figure(1)
for i in range(1,5):
    ax=fig.add_subplot(2,2,i)        
    ax.imshow(images_imgaug[i])
    ax.axis("off")
fig.suptitle("imgaug (backend skimage) "+str(np.round(dt,2))+"s")
plt.savefig("02_Brightness_imgaug.png")
plt.close(1)</code></pre>

![alt text](https://github.com/maikherbig/AIDeveloper/blob/master/Comparison%20of%20augmentation%20methods/art/02_Brightness_imgaug.png "02_Affine augmentation_Keras_ImageDataGenerator")  

### Brightness augmentation: AIDeveloper's aid_img.py  
<pre><code>t1 = time.time()
images_aid = aid_img.brightn_noise_augm_cv2(images,add_lower,add_upper,mult_lower,mult_upper,gaussnoise_mean,gaussnoise_scale)
t2 = time.time()
dt = t2-t1
print("AIDeveloper "+str(np.round(dt,2))+"s")

fig=plt.figure(1)
for i in range(1,5):
    ax=fig.add_subplot(2,2,i)        
    ax.imshow(images_aid[i])
    ax.axis("off")
fig.suptitle("AIDeveloper "+str(np.round(dt,2))+"s")
plt.savefig("02_Brightness_aid_img.png")
plt.close(1)</code></pre>

![alt text](https://github.com/maikherbig/AIDeveloper/blob/master/Comparison%20of%20augmentation%20methods/art/02_Brightness_aid_img.png "02_Affine augmentation_Keras_ImageDataGenerator")  

## Gaussian blur  
Define some parameters and the imgaug augmentation function:  
<pre><code>sigma_low = 0
sigma_high = 20

def imgaug_gaussnoise(images,sigma_high):
    seq = imgaug.augmenters.Sequential([
        imgaug.augmenters.blur.GaussianBlur(sigma=(sigma_low, sigma_high))
        ])
    images = seq(images=images)
    return images</code></pre>

### Gaussian blur: imgaug  
<pre><code>t1 = time.time()
images_imgaug = imgaug_gaussnoise(images,sigma_high)
t2 = time.time()
dt = t2-t1
print("imgaug "+str(np.round(dt,2))+"s")

fig=plt.figure(1)
for i in range(1,5):
    ax=fig.add_subplot(2,2,i)        
    ax.imshow(images_imgaug[i])
    ax.axis("off")
fig.suptitle("imgaug "+str(np.round(dt,2))+"s")
plt.savefig("03_Gaussblur_imgaug.png")
plt.close(1)</code></pre>

![alt text](https://github.com/maikherbig/AIDeveloper/blob/master/Comparison%20of%20augmentation%20methods/art/03_Gaussblur_imgaug.png "02_Affine augmentation_Keras_ImageDataGenerator")  

### Gaussian blur: AIDeveloper's aid_img.py  
<pre><code>t1 = time.time()
images_aid = aid_img.gauss_blur_cv(images,sigma_low,sigma_high)
t2 = time.time()
dt = t2-t1
print("AIDeveloper "+str(np.round(dt,2))+"s")

fig=plt.figure(1)
for i in range(1,5):
    ax=fig.add_subplot(2,2,i)        
    ax.imshow(images_aid[i])
    ax.axis("off")
fig.suptitle("AIDeveloper "+str(np.round(dt,2))+"s")
plt.savefig("03_Gaussblur_aid_img.png")
plt.close(1)</code></pre>

![alt text](https://github.com/maikherbig/AIDeveloper/blob/master/Comparison%20of%20augmentation%20methods/art/03_Gaussblur_aid_img.png "02_Affine augmentation_Keras_ImageDataGenerator")  

## Saturation/Hue augmentation  
Define some parameters and the imgaug augmentation function:  
<pre><code>hue_on = True
hue_low = 0.9
hue_high = 1.1
saturation_on = True
sat_low = 0.5
sat_high = 1.5

def imgaug_contrast(images,sigma_high):
    seq = imgaug.augmenters.Sequential([
        imgaug.augmenters.MultiplyHueAndSaturation(mul_saturation=(sat_low, sat_high),mul_hue=(hue_low, hue_high))
        ])
    images = seq(images=images)
    return images</code></pre>

### Saturation/Hue augmentation, imgaug  
<pre><code>t1 = time.time()
images_imgaug = imgaug_contrast(images,sigma_high)
t2 = time.time()
dt = t2-t1
print("imgaug "+str(np.round(dt,2))+"s")

fig=plt.figure(1)
for i in range(1,5):
    ax=fig.add_subplot(2,2,i)        
    ax.imshow(images_imgaug[i])
    ax.axis("off")
fig.suptitle("imgaug "+str(np.round(dt,2))+"s")
plt.savefig("04_Sat_Hue_imgaug.png")
plt.close(1)</code></pre>

![alt text](https://github.com/maikherbig/AIDeveloper/blob/master/Comparison%20of%20augmentation%20methods/art/04_Sat_Hue_imgaug.png "02_Affine augmentation_Keras_ImageDataGenerator")  

### Saturation/Hue augmentation, AIDeveloper's aid_img.py  
<pre><code>t1 = time.time()
images_aid = aid_img.satur_hue_augm_cv2(images,saturation_on,sat_low,sat_high,hue_on,hue_high-1)
t2 = time.time()
dt = t2-t1
print("AIDeveloper "+str(np.round(dt,2))+"s")

fig=plt.figure(1)
for i in range(1,5):
    ax=fig.add_subplot(2,2,i)        
    ax.imshow(images_aid[i])
    ax.axis("off")
fig.suptitle("AIDeveloper "+str(np.round(dt,2))+"s")
plt.savefig("04_Sat_Hue_aid_img.png")
plt.close(1)</code></pre>

![alt text](https://github.com/maikherbig/AIDeveloper/blob/master/Comparison%20of%20augmentation%20methods/art/04_Sat_Hue_aid_img.png "02_Affine augmentation_Keras_ImageDataGenerator")  

## System properties  
<pre><code>Print some information about the used system and python package versions
text = ""
text+="System used:\n"
text+="OS: "+platform.platform()+"\n"
text+="CPU: "+platform.processor()+"\n"
text+="Python package versions:\n"
text+="Keras: v"+keras.__version__+"\n"
text+="PIL: v"+PIL.__version__+"\n"
text+="TensorFlow: v"+tf.__version__+"\n"
text+="OpenCV: v"+cv2.__version__+"\n"</code></pre>  

Output:
<pre><code>System used:
OS: Windows-7-6.1.7601-SP1
CPU: Intel64 Family 6 Model 60 Stepping 3, GenuineIntel
Python package versions:
Keras: v2.2.5
PIL: v5.4.1
TensorFlow: v1.14.0
OpenCV: v4.1.1</code></pre> 


## Result, Summary  
This table provides an overview of the results. The table lists augmentation
operations and the required computational time for that operation when using
AIDevelopers aid_img.py (AID), imgaug, or ImageDataGenerator. All augmentation
operations were tested using 250 images of size 643x960 pixels.  
| Operation      | AID  | imgaug       | Keras ImageDataGenerator |
|----------------|------|--------------|--------------------------|
| Affine         | 1.9s | 2.1s / 50.4s | 38.2s                    |
| Brightness     | 3.1s | 11.3s        | 11.6s                    |
| Gaussian blur  | 0.4s | 3.1s         | NA                       |
| Saturation/Hue | 3.3s | 25.32s       | NA                       |

For each of the tested augmentation operations, the implementations of
AIDeveloper need the least time to finish. For the affine augmentation, 
AIDevelopers implementations are 20 times faster than Keras ImageDataGenerator.
This advantage in speed can mostly be accredited to efficient usage of OpenCV 
implementations. For the affine ("geometric") augmentation, the library imgaug 
provides scikit-image and OpenCV backends and the table shows that the same 
operation runs 24 times faster when using the OpenCV backend.
Currently, Keras ImageDataGenerator does not provide options for Gaussian blur 
or Saturation/Hue. Hence, these operations could only be compared between 
imgaug and AIDeveloper and the table shows that AIDevelopers implementation
are approximately 7 times faster.


