#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os,cv2
import numpy as np
import keras
import matplotlib.pyplot as plt
from sklearn import *
from PIL import *
import pandas as pd

from keras.models import model_from_json
from sklearn.utils import shuffle
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
#import theano
from keras import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from keras import backend as K
#K.set_image_dim_ordering('th')
from sklearn.metrics import multilabel_confusion_matrix

from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,Adam
from keras.layers.convolutional import *
import itertools
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


train_path=r"D:\dataset\project\birdclassification\train"
valid_path=r"D:\dataset\project\birdclassification\valid"
test_path=r"D:\dataset\project\birdclassification\test"


# In[14]:


train_batches=ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224), classes=['0.Black_footed_Albatross','1.Red_winged_Blackbird','2.Indigo_Bunting','3.Black_billed_Cuckoo','4.Pigeon_Guillemot','5.Ivory_Gull','6.Anna_Hummingbird','7.Pied_Kingfisher','8.Pacific_Loon','9.Ovenbird','10.American_Redstart','11.Savannah_Sparrow','12.Artic_Tern','13.Blue_headed_Vireo','14.Blue_winged_Warbler','15.Bohemian_Waxwing','16.Pileated_Woodpecker','17.Cardinal','18.Red_faced_cormonent','19.yellow_breasted_chat'],batch_size=10)
valid_batches=ImageDataGenerator().flow_from_directory(valid_path,target_size=(224,224), classes=['001.Black_footed_Albatross','010.Red_winged_Blackbird','014.Indigo_Bunting','031.Black_billed_Cuckoo','058.Pigeon_Guillemot','063.Ivory_Gull','067.Anna_Hummingbird','081.Pied_Kingfisher','086.Pacific_Loon','091.Mockingbird','092.Nighthawk','099.Ovenbird','108.White_necked_Raven','109.American_Redstart','127.Savannah_Sparrow','141.Artic_Tern','152.Blue_headed_Vireo','161.Blue_winged_Warbler','185.Bohemian_Waxwing','188.Pileated_Woodpecker'],batch_size=7)
test_batches=ImageDataGenerator().flow_from_directory(test_path,target_size=(224,224), classes=['001.Black_footed_Albatross','010.Red_winged_Blackbird','014.Indigo_Bunting','031.Black_billed_Cuckoo','058.Pigeon_Guillemot','063.Ivory_Gull','067.Anna_Hummingbird','081.Pied_Kingfisher','086.Pacific_Loon','091.Mockingbird','092.Nighthawk','099.Ovenbird','108.White_necked_Raven','109.American_Redstart','127.Savannah_Sparrow','141.Artic_Tern','152.Blue_headed_Vireo','161.Blue_winged_Warbler','185.Bohemian_Waxwing','188.Pileated_Woodpecker'],batch_size=5)


# In[15]:


def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims=np.array(ims).astype(np.uint8)
        if(ims.shape[-1]!=3):
            ims=ims.transpose((0,2,3,1))
    f=plt.figure(figsize=figsize)
    cols=len(ims)//rows if len(ims)%2 == 0 else len(ims)//rows +1
    for i in range(len(ims)):
        sp=f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[1],fontsize=10)
        plt.imshow(ims[1],interpolation=None if interp else 'none')
        


# In[16]:


imgs, labels=next(train_batches)


# In[17]:


plots(imgs, titles=labels)


# In[18]:


#Building and training Cnn model
labels=['0.Black_footed_Albatross','1.Red_winged_Blackbird','2.Indigo_Bunting','3.Black_billed_Cuckoo','4.Pigeon_Guillemot','5.Ivory_Gull','6.Anna_Hummingbird','7.Pied_Kingfisher','8.Pacific_Loon','9.Ovenbird','10.American_Redstart','11.Savannah_Sparrow','12.Artic_Tern','13.Blue_headed_Vireo','14.Blue_winged_Warbler','15.Bohemian_Waxwing','16.Pileated_Woodpecker','17.Cardinal','18.Red_faced_cormonent','19.yellow breasted chat']


# In[19]:


model=Sequential([
                Conv2D(32, (3, 3), input_shape=(224,224,3),activation='relu'),
                  Flatten(),
                  Dense(20, activation='softmax'),
                 ])


# In[20]:


model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:





# In[21]:


model.fit_generator(train_batches, steps_per_epoch=119,
                    validation_data=valid_batches, validation_steps=54, epochs=5, verbose=2)


# In[ ]:




model.save_weights("main_model.h5")
print("Saved model to disk")
 
# later...
 


# In[ ]:



#load_model
model.load_weights("C://Users//Lenovo//main_model.h5")
print("Loaded model from disk")


# In[ ]:


test_imgs, test_labels = next(test_batches)
plots(test_imgs, titles=test_labels)


# In[ ]:


test_labels = test_labels
np.array(test_labels)


# In[ ]:


predictions= model.predict_generator(test_batches, steps=20, verbose=1)


# In[ ]:


np.array(predictions)


# In[ ]:


for i in predictions:
    n=np.argmax(i)
    print(n+1)


# In[ ]:


#cm=multilabel_confusion_matrix(test_labels, predictions,labels=['001.Black_footed_Albatross','010.Red_winged_Blackbird','014.Indigo_Bunting','031.Black_billed_Cuckoo','058.Pigeon_Guillemot','063.Ivory_Gull','067.Anna_Hummingbird','081.Pied_Kingfisher','086.Pacific_Loon','091.Mockingbird','092.Nighthawk','099.Ovenbird','108.White_necked_Raven','109.American_Redstart','127.Savannah_Sparrow','141.Artic_Tern','152.Blue_headed_Vireo','161.Blue_winged_Warbler','185.Bohemian_Waxwing','188.Pileated_Woodpecker'])


# In[ ]:


#score = model.evaluate(test_labels, predictions)


# In[ ]:


#tuning model using vgg16


# In[23]:


vgg16_model = keras.applications.vgg16.VGG16()


# In[24]:


vgg16_model.summary()


# In[ ]:


type(vgg16_model)


# In[ ]:


model= Sequential()
for layer in vgg16_model.layers:
    model.add(layer)


# In[ ]:


model.summary()


# In[ ]:


model.layers.pop()


# In[ ]:


for layer in model.layers:
    layers.trainable=False


# In[ ]:


model.add(Dense(20, activation='softmax'))


# In[ ]:


model.summary()


# In[ ]:


#train fine-tuned model


# In[25]:


model.compile(Adam(lr=0.000001), loss='categorical_crossentropy', metrics=['accuracy'])


# In[27]:


model.fit_generator(train_batches, steps_per_epoch=119,
                      validation_data=valid_batches, validation_steps=54, epochs=150, verbose=2)


# In[28]:


model.save_weights("main_model.h5")
print("Saved model to disk")


# In[29]:


model.load_weights("C://Users//Lenovo//main_model.h5")
print("Loaded model from disk")


# In[30]:


test_imgs, test_labels = next(test_batches)
plots(test_imgs, titles=test_labels)


# In[31]:


test_labels = test_labels
np.array(test_labels)


# In[ ]:


predictions= model.predict_generator(test_batches, steps=20, verbose=-1)


# In[ ]:


np.array(predictions)


# In[ ]:


for i in predictions:
    n=np.argmax(i)
    print(n+1)


# In[ ]:




