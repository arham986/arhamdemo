
# coding: utf-8

# In[2]:


import numpy as np
np.random.seed(1000)
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split


# In[3]:


import numpy as np
np.random.seed(1000)
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split


# In[4]:


import keras
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.models import Sequential


# In[5]:


import os
import cv2
from PIL import Image


# In[6]:


DATA_DIR = 'C:\\Users\\Lenovo\\Desktop\\pythonDS\\cell_images\\'
SIZE = 64
dataset = []
label = []


# In[8]:


parasitized_images = os.listdir(DATA_DIR + 'Parasitized\\')


# In[9]:


len(parasitized_images)


# In[10]:


for i, image_name in enumerate(parasitized_images):
    try:
        if (image_name.split('.')[1] == 'png'):
            image = cv2.imread(DATA_DIR + 'Parasitized\\' + image_name)
            image = Image.fromarray(image, 'RGB')
            image = image.resize((SIZE, SIZE))
            dataset.append(np.array(image))
            label.append(0)
    except Exception:
        print("Could not read image {} with name {}".format(i, image_name))


# In[12]:


uninfected_images = os.listdir(DATA_DIR + 'Uninfected\\')


# In[13]:



for i, image_name in enumerate(uninfected_images):
    try:
        if (image_name.split('.')[1] == 'png'):
            image = cv2.imread(DATA_DIR + 'Uninfected\\' + image_name)
            image = Image.fromarray(image, 'RGB')
            image = image.resize((SIZE, SIZE))
            dataset.append(np.array(image))
            label.append(1)
    except Exception:
        print("Could not read image {} with name {}".format(i, image_name))


# In[27]:


plt.figure(figsize = (20, 12))
for index, image_index in enumerate(np.random.randint(len(parasitized_images), size = 5)):
    plt.subplot(1, 5, index+1)
    plt.imshow(dataset[image_index])


# In[29]:


plt.figure(figsize = (20, 12))
for index, image_index in enumerate(np.random.randint(len(uninfected_images), size = 5)):
    plt.subplot(1, 5, index+1)
    plt.imshow(dataset[len(parasitized_images) + image_index])


# In[17]:


plt.figure(figsize = (20, 12))
x=1
for img in range(0,5):
    plt.subplot(1,5,x)
    plt.imshow(dataset[img])
    x+=1


# In[18]:


model=Sequential()
model.add(Convolution2D(32, (3,3) , input_shape=(SIZE,SIZE,3)  , activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), data_format="channels_last"))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.2))
model.add(Convolution2D(32, (3,3) , activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), data_format="channels_last"))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units=512,activation="relu"))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.2))
model.add(Dense(units=256,activation="relu"))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.2))
model.add(Dense(units=2,activation="softmax"))


# In[19]:


model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])


# In[20]:


print(model.summary())


# In[23]:


from keras.utils  import to_categorical
xtrain,xtest,ytrain,ytest=train_test_split(dataset,to_categorical(np.array(label)),test_size=0.2,random_state=0)


# In[24]:


model.fit(np.array(xtrain),ytrain,batch_size=64,epochs=50,verbose=2,validation_split=0.1,shuffle=False)


# In[26]:


print("Test_Accuracy: {:.2f}%".format(model.evaluate(np.array(xtest), np.array(ytest))[1]*100))


# In[27]:


from keras.preprocessing.image import ImageDataGenerator





train_generator = ImageDataGenerator(rescale = 1/255,
                                     zoom_range = 0.3,
                                     horizontal_flip = True,
                                     rotation_range = 30)

test_generator = ImageDataGenerator(rescale = 1/255)

train_generator = train_generator.flow(np.array(xtrain),
                                       ytrain,
                                       batch_size = 64,
                                       shuffle = False)

test_generator = test_generator.flow(np.array(xtest),
                                     ytest,
                                     batch_size = 64,
                                     shuffle = False)


# In[28]:


history = model.fit_generator(train_generator,
                                   steps_per_epoch = len(xtrain)/64,
                                   epochs = 50,
                                   shuffle = False)


# In[ ]:


print("Test_Accuracy(after augmentation): {:.2f}%".format(model.evaluate_generator(test_generator, steps = len(xtest), verbose = 1)[1]*100))

