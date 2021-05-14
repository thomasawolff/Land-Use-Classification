#from __future__ import absolute_import, division, print_function, unicode_literals

import os
import csv
import glob
import time 
import pathlib
import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import compat
from keras.preprocessing import image
import matplotlib.image as mpimg
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import Xception
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import TensorBoard
from keras.preprocessing.text import Tokenizer
#from tensorflow.keras.applications import MobileNet
from keras.applications.resnet import ResNet152
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
from keras.applications.mobilenet import preprocess_input
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator



path = r'E:\DeepLearningImages\Deep Learning Code and Images'
name = 'LandUsePredict-cnn-64x2-{}'.format(time.time())
tensorboard = TensorBoard(log_dir=path+'\\logs\\{}'.format(name),
                          histogram_freq=1,
                          write_images=True)


############## This model uses tensorflow-GPU 2.3.0 #################

#path = (r'https://drive.google.com/drive/folders/1LS7mECdPTtcCmSUcOVKMOnsCndwHx-Dh')
train_dir = os.path.join(path, 'train')
val_dir = os.path.join(path, 'validation')
test_dir = os.path.join(path, 'test')
train_dir = pathlib.Path(train_dir)
val_dir = pathlib.Path(val_dir)
test_dir = pathlib.Path(test_dir)

##train_image_count = len(list(train_dir.glob('*/*.tif')))
##val_image_count = len(list(val_dir.glob('*/*.tif')))
##test_image_count = len(list(test_dir.glob('*/*.tif')))



def imageDimensions():
    dim1 = []
    dim2 = []

    for root, dirs, files in os.walk(train_dir, topdown=False):
        for name in files:
            try:
                img = imread(os.path.join(root, name))
                d1,d2,colors = img.shape
                #print(d1,d2)
                dim1.append(d1)
                dim2.append(d2)
            except: pass
    #print(np.mean(dim1))
    #print(np.mean(dim2))

    #sns.jointplot(x=dim1,y=dim2)
    #plt.show()
    return int(np.mean(dim1)),int(np.mean(dim2))

#print(imageDimensions()[0])


class dataSetupRun(object):

    def __init__(self):

        self.preTrainedModel = 'imagenet' # pretrained model used for classification
        self.denseActivationFunc = 'relu' # activation function used with in dense layers
        self.predsActivationFunc = 'softmax' # activation function used to measure loss
        self.optimizerFunc = 'adagrad' # optimizer for backpropagation
        self.classMode = 'categorical' # the kind of machine learning to be done
        self.batch_size = 8 # the number of images included processed at once for classification
        self.img_height = 224 #imageDimensions()[0]  
        self.img_width = 224 #imageDimensions()[0]
        self.total_train = len(list(train_dir.glob('*/*.tif')))
        self.total_val = len(list(val_dir.glob('*/*.tif')))
        self.architecture = ResNet152
        self.epochs = 2000 # the number of iterations through training set
        self.bands = 3 # color image has 3 color bands, red, green, blue
    



    @tf.function(experimental_compile=True)
    def arrangeData(self):

        # training image generator. in this generator I am modifying the training images each iteration
        # so as to prevent overfitting during training. validation images are not modified.
        train_Image_generator = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=6,
                                   width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                                   horizontal_flip=True, fill_mode='nearest')


        # validation image generator. these images are not modified.
        val_image_generator = ImageDataGenerator(rescale=1./255)


        # test image generator. these images are not modified.
        test_Image_generator = ImageDataGenerator(rescale=1./255)

        # generating the training images and converting them into data usable by the classification algorithm
        self.train_data_gen = train_Image_generator.flow_from_directory(batch_size=self.batch_size,
                                                        directory=train_dir,
                                                        shuffle=True, # images will be shuffled each iteration
                                                        color_mode="rgb",
                                                        target_size=(self.img_height, self.img_width),
                                                        class_mode=self.classMode)


        # generating the validation images
        self.validation_data_gen = val_image_generator.flow_from_directory(batch_size=self.batch_size,
                                                        directory=val_dir,
                                                        color_mode="rgb",
                                                        target_size=(self.img_height, self.img_width),
                                                        class_mode=self.classMode)


        # generating the test images which are seperate from train and validation images
        # which the algorithm will have not seen
        self.test_data_gen = test_Image_generator.flow_from_directory(directory=test_dir,
                                                        color_mode="rgb",
                                                        target_size=(self.img_height, self.img_width),
                                                        class_mode=self.classMode,
                                                        shuffle=False)
                                                        #save_to_dir = path+'\\testImagesPredicted',
                                                        #save_format = 'jpeg')


    
    def modelSetupRun(self):
        self.arrangeData()

        # getting the number of labels in image data
        labels = (self.train_data_gen.class_indices)
        labels = dict((v,k) for k,v in labels.items())

        # assigning the pre-trained model MobileNet to the variable base_model
        base_model=self.architecture(input_shape=(self.img_height,self.img_width,self.bands),\
                             weights=self.preTrainedModel,include_top=False)

        denseLayers = base_model.output # brining in the output from the base_model into dense layers
        denseLayers = Flatten()(denseLayers)

        preds = Dense(len(labels),activation = self.predsActivationFunc)(denseLayers) #final dense layer with softmax activation

        #self.model.trainable = False # setting the pretrained model to be trainable
        for layer in base_model.layers:
            layer.trainable = True

        pd.set_option('max_colwidth', None)
        layers = [(layer, layer.name, layer.trainable) for layer in base_model.layers]
        print(pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable']))


        self.model = Model(inputs=base_model.input, outputs=preds)
        self.model.compile(optimizer=self.optimizerFunc, # compiling the model using adagrad optimizer
                      loss=CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        #model.summary()
        return self.model



    def runTrainCompile(self):
        self.modelSetupRun()
        early_stop = EarlyStopping(monitor='val_loss',patience=2)
        
        # training the model and doing initial evalution using validation data
        
        self.history = self.model.fit(
            self.train_data_gen,
            #steps_per_epoch=self.total_train // self.batch_size,
            epochs=self.epochs,
            validation_data=self.validation_data_gen,
            #validation_steps=self.total_val // self.batch_size,
            callbacks = [tensorboard, early_stop]
            )

        self.model.save(r'E:\DeepLearningImages\Deep Learning Code and Images\savedClassModel.h5')
        losses = pd.DataFrame(self.model.history.history)
        losses[['loss','val_loss']].plot()
        plt.show()



    def performanceViz(self):
        self.runTrainCompile()

        history_dict = self.history.history
        acc = history_dict['acc']
        val_acc = history_dict['val_acc']
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']

        compat.v1.RunOptions(report_tensor_allocations_upon_oom = True)

        print(acc)
        epochs_range = range(self.epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()
    



    def testDataPredictionsProbs(self,index_):
        preds = []
        labelsList = []
        self.arrangeData()
        savedModel = tf.keras.models.load_model('savedClassModel.h5')
        
        loss,acc = savedModel.evaluate(self.test_data_gen)
        predict = savedModel.predict(self.test_data_gen)
        labels = (self.test_data_gen.class_indices)
        labels = dict((v,k) for k,v in labels.items())
        predicted_class_indices=np.argmax(predict,axis=1)
        self.predictions = [labels[k] for k in predicted_class_indices]
        self.filenames=self.test_data_gen.filenames

        labelsPredDict = dict(zip(self.filenames,self.predictions))
        for key,value in labelsPredDict.items() :
            preds.append([key,value])
            
        print(preds[index_])
        print(predict[index_])
        my_cmap = plt.get_cmap('tab20c')

        plt.figure(figsize=(15,10))
        plt.tight_layout()
        plot = plt.bar(labels.values(),predict[index_],data=predict[index_],log=True,color=my_cmap.colors)
        plt.xticks([])
        plt.ylabel('Log Probabilities')
        plt.title('Class prediction probabilities: '+str(preds[index_]))
        plt.legend(plot,[i for i in labels.values()],loc="upper left")
        plt.show()




    def testDataPredictionsWrite(self):
        self.arrangeData()
        savedModel = tf.keras.models.load_model('savedClassModel.h5')
        
        loss = savedModel.evaluate(self.test_data_gen)
        predict = savedModel.predict(self.test_data_gen)
        predicted_class_indices=np.argmax(predict,axis=1)
        
        labels = (self.test_data_gen.class_indices)
        labels = dict((v,k) for k,v in labels.items())
        self.predictions = [labels[k] for k in predicted_class_indices]

        self.filenames=self.test_data_gen.filenames
        results=pd.DataFrame({'Filename':self.filenames,'Predictions':self.predictions})
        print(results)

        results.to_csv('CNN_Results_Output.csv', sep='\t')


    
        
#print(imageDimensions())
#go = dataSetupRun()
#go.arrangeData()
#go.modelSetupRun()
#go.runTrainCompile()
#go.testDataPredictionsProbs(356)
#go.testDataPredictionsWrite()
#go.performanceViz()
















