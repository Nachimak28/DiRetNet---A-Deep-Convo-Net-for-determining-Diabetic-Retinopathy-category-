import numpy as np
import tensorflow as tf
import keras
#from keras import layers
#from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input

#from IPython.display import SVG
#from keras.utils.vis_utils import model_to_dot
#from keras.utils import plot_model
#from resnets_utils import *
#from keras.initializers import glorot_uniform
#import scipy.misc
from matplotlib.pyplot import imshow
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
#import os
#from scipy.misc import imread, imsave, imresize
#from sklearn.utils import shuffle
#from sklearn.cross_validation import train_test_split
#from sklearn import preprocessing
#import glob
#import scipy.misc
import time
#from tkinter import filedialog
#from tkinter import *
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
tic = time.time()
size = 299
model = load_model('DR_model.h5')

#root = Tk()
#root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
#print (root.filename)

#img_path = root.filename
#img = image.load_img(img_path, target_size = (size,size))
#imshow(img)
#
#x = image.img_to_array(img)
#x = np.expand_dims(x, axis = 0)
#x = preprocess_input(x)
#op = (dr_model.predict(x))*100
#print(np.round_(op,4))
#print(type(dr_model.predict(x)))
#print(op.shape)
#print('Catrgory 0: ' + str(op[0][0]))
#print('Catrgory 1: ' + str(op[0][1]))
#print('Catrgory 3: ' + str(op[0][2]))
#print('Catrgory 4: ' + str(op[0][3]))
#print('Catrgory 4: ' + str(op[0][4]))

test_data = ImageDataGenerator(rescale = 1./255,samplewise_std_normalization=True)       #preprocessing and real tiime data augmentation of test set images
X_pred = test_data.flow_from_directory('C:/Users/nachiket/Desktop/SEM_8/BE_project/Codes/predict',shuffle = False,target_size = (size, size),batch_size = 64,class_mode = None)       #Creates a DirectoryIterator object  for getting images from the directory specified with images in sub directories 0,1,2,3,4 for train set 
op = model.predict_generator(X_pred)     #predicting for single image
#plot_model(dr_model, to_file='DRModel.png')
#SVG(model_to_dot(dr_model).create(prog='dot', format='svg'))
#print(np.round_(op*100,4))

y_pred = []
for i in range(len(op)):
    for j in range(len(op[i])):
        if op[i][j] == max(op[i]):
            if j == 2 or j == 3:
                #print('Category ',j+1)
                y_pred.append(j+1)
            else:
                #print('Category ',j)
                y_pred.append(j)
y_pred = np.array(y_pred)
#np.save('y_pred.npy', y_pred)
print('Predicted Output\t\tConfidence(%)\t\tActual')
a = ['0','1','3','4']

for i in range(len(op)):
    print('Category '+str(y_pred[i])+'\t\t\t'+str(max(op[i])*100)+'\t'+str(a[i]))
toc = time.time()
print("Time taken= " + str((toc-tic)) + "s")
