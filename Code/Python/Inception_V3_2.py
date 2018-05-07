from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model, load_model
from keras import optimizers
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers 
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input
from sklearn.metrics import roc_auc_score
from tkinter import filedialog
from tkinter import *
import numpy as np
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import time
import matplotlib.pyplot as plt
#from keras.applications.imagenet_utils import preprocess_input, decode_predictions
tic = time.time()
size = 299
#epochs = 50
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

a = input("Already trained the model? Y/n: ")           #asking user if the model is already trained or not, If 'n' then the training starts, if 'Y' then pretrained model is loaded 

if a == 'n':
    train_data = ImageDataGenerator(rescale = 1./255, horizontal_flip = True, vertical_flip = True,samplewise_std_normalization=True, shear_range = 0.2, zoom_range = 0.1)     #preprocessing and real tiime data augmentation of training set images

    test_data = ImageDataGenerator(rescale = 1./255, horizontal_flip = True, vertical_flip = True,samplewise_std_normalization=True, shear_range = 0.2, zoom_range = 0.1)       #preprocessing and real tiime data augmentation of test set images

    X_train = train_data.flow_from_directory('C:/Users/nachiket/Desktop/SEM_8/BE_project/final_dataset/Final_dataset_train',target_size = (size, size),batch_size = 64,class_mode = 'categorical')    #Creates a DirectoryIterator object  for getting images from the directory specified with images in sub directories 0,1,2,3,4 for train set 
    label_map = (X_train.class_indices)
    print(label_map)

    X_test = test_data.flow_from_directory('C:/Users/nachiket/Desktop/SEM_8/BE_project/final_dataset/Final_dataset_test',target_size = (size, size),batch_size = 64,class_mode = 'categorical')       #Creates a DirectoryIterator object  for getting images from the directory specified with images in sub directories 0,1,2,3,4 for train set 
    label_map1 = (X_test.class_indices)
    print(label_map1)
   
    input_shape = (size, size, 3)
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu',kernel_regularizer=regularizers.l2(0.01), name = 'FC_sftm')(x)
    x = Dropout(0.25)(x)


    # and a logistic layer --  we have 5 classes
    predictions = Dense(5, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    #model.load_model('tp2.model')
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999), loss='categorical_crossentropy', metrics = ['accuracy'])

# train the model on the new data for a few epochs
    hist1 = model.fit_generator(X_train,steps_per_epoch = (21667/64),epochs = 4,validation_data = X_test,validation_steps = 40, initial_epoch = 4)            #steps per epoch = number of samples/batch_size
    
    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from inception V3. We will freeze the bottom N layers
    # and train the remaining top layers.

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    for i, layer in enumerate(base_model.layers):
       print(i, layer.name)

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in model.layers[:249]:
       layer.trainable = False
    for layer in model.layers[249:]:
       layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    #from keras.optimizers import ADAM
    model.compile(optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999), loss='categorical_crossentropy', metrics = ['accuracy'])

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    hist2 = model.fit_generator(X_train,
                           steps_per_epoch = (21667/64),
                           epochs = 10,
                           validation_data = X_test,
                           validation_steps = 40, initial_epoch = 2, callbacks = [keras.callbacks.TensorBoard(log_dir = 'C:/Users/nachiket/Desktop/SEM_8/BE_project/logging',histogram_freq = 2, batch_size = 64, write_graph = True, write_grade = True, write_images = True)])


    
    model.save('tp3.h5')
    
    def plot_training(history):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))

        plt.plot(epochs, acc, 'r.')
        plt.plot(epochs, val_acc, 'r')
        plt.title('Training and validation accuracy')

        plt.figure()
        plt.plot(epochs, loss, 'r.')
        plt.plot(epochs, val_loss, 'r-')
        plt.title('Training and validation loss')
        plt.show()

    plot_training(hist2)

elif a == 'Y':
    model = load_model('Inception_retrained.h5')                 #loading a saved model
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    print('Weights and model loaded')

#root = Tk()             #for GUI of file dialog box
#root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))  #opening file dialog box for choosing single image for prediction
#print (root.filename)

#img_path = root.filename
#img = image.load_img(img_path, target_size = (299,299))
#imshow(img)

#x = image.img_to_array(img)
#x = x/255.0
#x = np.expand_dims(x, axis = 0)
#print(x.shape)
#x = preprocess_input(x)
test_data = ImageDataGenerator(rescale = 1./255,samplewise_std_normalization=True)       #preprocessing and real tiime data augmentation of test set images
X_pred = test_data.flow_from_directory('C:/Users/nachiket/Desktop/SEM_8/BE_project/Codes/predict',shuffle = False,target_size = (size, size),batch_size = 64,class_mode = None)       #Creates a DirectoryIterator object  for getting images from the directory specified with images in sub directories 0,1,2,3,4 for train set 
op = model.predict_generator(X_pred)     #predicting for single image
#op = model.predict(x)
#print(op)
#print(np.round_(op,4))
#print(type(model.predict(x)))
#print(op.shape)
#print('Catrgory 0: ' + str(op[0][0]))
#print('Catrgory 1: ' + str(op[0][1]))
#print('Catrgory 2: ' + str(op[0][2]))
#print('Catrgory 3: ' + str(op[0][3]))
#print('Catrgory 4: ' + str(op[0][4]))

#print(op[0][0]+ op[0][1]+op[0][2]+op[0][3]+op[0][4])
#plot_model(model, to_file='DRModelInception.png')
toc = time.time()
print("Time taken= " + str((toc-tic)) + "s")
