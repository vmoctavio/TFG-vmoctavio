# -----------------------------------------------------------------------------------------------
# Programa: TFGVOP_DenseNet201
# Autor: vmoctavio
# Fecha creación: 22/09/2019
# Descripción: Proceso para generar un Dataset sintético a partir de una estructura
#                   directory_root + plant_disease_folder
#              generando
#                   directory_dest + plant_disease_folder
# -----------------------------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------------------------
# Imports necesarios
# -----------------------------------------------------------------------------------------------
#
# Convertir una imagen en una matriz NumPy 
from keras.preprocessing.image import img_to_array
#
# Devuelve una lista que contiene el nombre de las entradas en el directorio de la ruta del parámetro.
from os import listdir
#
# Manejo de arrays 
import numpy as np
#
# Usar comandos del sistema operativo
import os
#
# Para manipular fechas y horas
import datetime

###### desde aquí
#Para qué sirve el cv2
import cv2
#
#Para qué sirve el datetime
import datetime
#
# Para qué sirve el matplotlib.pyplot
import matplotlib.pyplot as plt
#
# Para qué sirve el pickle
import pickle
#
# Para qué sirve el
from keras import backend as K
#
# Para qué sirve el
from keras.optimizers import Adam
#
# Para qué sirve el
from keras.layers import Conv2D
#
# Para qué sirve el
from keras.layers import MaxPooling2D
#
# Para qué sirve el
from keras.layers import Activation, Flatten, Dropout, Dense
#
# Para qué sirve el
from keras.layers.normalization import BatchNormalization
#
# Para qué sirve el
from keras.models import Sequential
#
# Para qué sirve el
from keras.preprocessing import image
#
# Para qué sirve el
from sklearn.model_selection import train_test_split#
#
# Para qué sirve el
from sklearn.preprocessing import LabelBinarizer
#
# Para qué sirve el
from sklearn.preprocessing import MultiLabelBinarizer
#


import os
import os.path
import shutil

# Para qué sirve el
import tensorflow as tf

#from tensorflow.keras import layers

import tensorboard as tb
import keras

from tensorflow.keras.applications import DenseNet201




print("Versión Tensorflow",tf.__version__)
print("Versión tensorboard",tb.__version__)
print("Versión keras",keras.__version__)



###### hasta aquí

#
# -----------------------------------------------------------------------------------------------
# Inicialización de variables
# -----------------------------------------------------------------------------------------------
#
# Directorio de trabajo donde está el dataset
directory_root = '/Users/vmoctavio/Downloads/keras-tutorial-dataaugmentation/Dataset_vid_prueba_dest/'
# Directorio de trabajo donde se generan los logs de salida del proceso
directory_log = '/Users/vmoctavio/Downloads/keras-tutorial/logs/'
# Número de veces que se procesará el dataset de entrada
EPOCHS = 25
# para qué es esto
INIT_LR = 1e-3
# para qué es esto
BS = 32
# para qué es esto
image_size = 0
# para qué es esto
default_image_size = tuple((256, 256))
# para qué es esto
width=256
# para qué es esto
height=256
# para qué es esto
depth=3
#
# -----------------------------------------------------------------------------------------------
# Función para convertir una imagen en array
# -----------------------------------------------------------------------------------------------
#
def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None:
            image = cv2.resize(image, default_image_size)
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None
#
# -----------------------------------------------------------------------------------------------
# Proceso principal: bucle que lee las carpetas del directorio del parámetros
#                    y los archivos contenidos en ellas, 
#                    y genera el array de imágenes y etiquetas
# -----------------------------------------------------------------------------------------------
#
# inicializa arrays de imágenes y etiquetas
image_list, label_list = [], []
#
try:
    print("[INFO]", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Inicio del proceso TFGVOP_DenseNet201 ")
    root_dir = listdir(directory_root)      # Genera una lista con los directorios existentes
#
    for directory in root_dir :
        # elimina .DS_Store de la lista en caso de existir
        if directory == ".DS_Store" :
            root_dir.remove(directory)
# 
    plant_disease_folder_list = listdir(f"{directory_root}") # Genera una lista con los directorios existentes
    for disease_folder in plant_disease_folder_list:
        # elimina .DS_Store de la lista en caso de existir
        if disease_folder == ".DS_Store" :
            plant_disease_folder_list.remove(disease_folder)
#
    for plant_disease_folder in plant_disease_folder_list:  # Nos recorremos todos los directorios existentes
        print(f"[INFO]", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Procesando el directorio ... ",plant_disease_folder)
#
        total_images_origen = 0 # contador imágenes por cada directorio
#        
        # Genera una lista con todos los archivos de un directorio concreto
        plant_disease_image_list = listdir(f"{directory_root}/{plant_disease_folder}/")
# 
        for single_plant_disease_image in plant_disease_image_list:  # Nos recorremos todos los archivos de un directorio concreto
            # elimina .DS_Store de la lista en caso de existir
            if single_plant_disease_image == ".DS_Store" :
                plant_disease_image_list.remove(single_plant_disease_image)

        for image in plant_disease_image_list: # Nos recorremos todos los archivos de un directorio concreto
            image_directory = f"{directory_root}/{plant_disease_folder}/{image}"
            # Si el archivo a tratar tiene extensión jpg o JPG se ejecuta "dataaugmentation" 
            if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:
#               Carga lista de imágenes y etiquetas
                image_list.append(convert_image_to_array(image_directory))  # Convierte imagen a array
                label_list.append(plant_disease_folder)
                total_images_origen = total_images_origen + 1 #  contador imágenes por cada directorio
#
        print("[INFO]", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Total imágenes del directorio ... ",plant_disease_folder,total_images_origen)
#
    print("[INFO]", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Proceso de carga de imágenes completado")
except Exception as e:
    print(f"Error : {e}")
#
# -----------------------------------------------------------------------------------------------
# Obtener el tamño de la imagen
# -----------------------------------------------------------------------------------------------
#
image_size = len(image_list)
#
# -----------------------------------------------------------------------------------------------
# Transforma las etiquetas de las imágenes mediante LabelBinarizer
# -----------------------------------------------------------------------------------------------
#
label_binarizer = LabelBinarizer()
image_labels = label_binarizer.fit_transform(label_list)
pickle.dump(label_binarizer,open('label_transform.pkl', 'wb'))    # esto creo que podría sobrar, no sé para qué es.
n_classes = len(label_binarizer.classes_)
#
print("[INFO]", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Etiquetas ... ",label_binarizer.classes_)
#
# -----------------------------------------------------------------------------------------------
# xxxxxx
# -----------------------------------------------------------------------------------------------
#
np_image_list = np.array(image_list, dtype=np.float16) / 225.0
#
# -----------------------------------------------------------------------------------------------
# xxxxxx
# -----------------------------------------------------------------------------------------------
#
print("[INFO]", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Dividiendo datos en train y test...")
x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.2, random_state = 42)

# pensar si el test_size y demás de pone como parámetros....

log_dir=directory_log + "/pruebavidcon/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#input_shape = (224, 224, 3)

model = tf.keras.Sequential()

inputShape = (height, width, depth)
chanDim = -1
if K.image_data_format() == "channels_first":
    inputShape = (depth, height, width)
    chanDim = 1
    
print('inputShape: ',inputShape)    
print('n_classes: ',n_classes)    

def create_DenseNet201():  
    model = DenseNet201(include_top=True, weights=None, input_tensor=None, input_shape=inputShape, pooling=None, classes=n_classes)
    return model

print('DenseNet201: ')
DenseNet201_model = create_DenseNet201()  
DenseNet201_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc', 'mse'])  

DenseNet201_model.summary()

DenseNet201 = DenseNet201_model.fit(x=x_train, y=y_train, batch_size=32, epochs=10, verbose=1, validation_data=(x_test, y_test), shuffle=True, callbacks=[tensorboard_callback])  

plt.figure(0)  
plt.plot(DenseNet201.history['acc'],'r')  
plt.plot(DenseNet201.history['val_acc'],'g')  
plt.xticks(np.arange(0, 11, 2.0))  
plt.rcParams['figure.figsize'] = (8, 6)  
plt.xlabel("Num of Epochs")  
plt.ylabel("Accuracy")  
plt.title("Training Accuracy vs Validation Accuracy DenseNet201")  
plt.legend(['train','validation'])

plt.figure(1)  
plt.plot(DenseNet201.history['loss'],'r')  
plt.plot(DenseNet201.history['val_loss'],'g')  
plt.xticks(np.arange(0, 11, 2.0))  
plt.rcParams['figure.figsize'] = (8, 6)  
plt.xlabel("Num of Epochs")  
plt.ylabel("Loss")  
plt.title("Training Loss vs Validation Loss DenseNet201")  
plt.legend(['train','validation'])

plt.show() 

# -----------------------------------------------------------------------------------------------
# xxxxxxx esto es posible que se tenga que borrar es para ver qué sale
# -----------------------------------------------------------------------------------------------
#
# XXXXXXXX
print("DenseNet201.history.keys(): ",DenseNet201.history.keys())

#
# -----------------------------------------------------------------------------------------------
# Model Accuracy
# -----------------------------------------------------------------------------------------------
#
# XXXXXXXX
print("[INFO] Calculating model accuracy")
scores = DenseNet201_model.evaluate(x_test, y_test)
print(f"Test Accuracy: {scores[1]*100}")

print(f"el 0: {scores[0]}")

# ver qué es el cero qu eel número no me cuadra y lo que imprime antes de loss: tampoco sé qué dato es para interpretarlo

print("ACCURACY :")
print(f"Training accuracy : {DenseNet201.history['acc'][-1]}")
print(f"Validation accuracy : {DenseNet201.history['val_acc'][-1]}")
    
print("\nLOSS :")
print(f"Training categorical crossentropy loss : {DenseNet201.history['loss'][-1]}")
print(f"Validation categorical crossentropy loss : {DenseNet201.history['val_loss'][-1]}")

# 6.4 Learning rate ¶

# desde aquí da error hasta aquí
def plot_lr(history):
    lr=1e-4
    fig, ax = plt.subplots(figsize=(7, 5))
    
# Plot learning rate
    ax.plot(DenseNet201.history['lr'])
    ax.set_title('Learning rate evolution')
    ax.set_ylabel('Learning rate value')
    ax.set_xlabel('Epochs')
    ax.legend(['Train'], loc='upper right')
    
plot_lr(DenseNet201)  
# hasta aquí da error

# esto tampoco funciona
from keras.utils import to_categorical
y_train_cat = to_categorical(y_train)
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred,axis=1)

LABELS = ['Blackrot', 'Esca', 'Healthy', 'Leaf_blight']

def plot_pictures(X, y, nb_rows=3, nb_cols=3, figsize=(14, 14)):
    # Set up the grid
    fig, ax = plt.subplots(nb_rows, nb_cols, figsize=figsize, gridspec_kw=None)
    fig.subplots_adjust(wspace=0.4, hspace=0.4)

    for i in range(0, nb_rows):
        for j in range(0, nb_cols):
            index = np.random.randint(0, X.shape[0])
    
            # Hide grid
            ax[i, j].grid(False)
            ax[i, j].axis('off')
            
            # Plot picture on grid
            ax[i, j].imshow(X[index].astype(np.int))
            ax[i, j].set_title(f"{LABELS[np.where(y[index] == 1)[0][0]]}")
            
plot_pictures(x_test, to_categorical(y_pred))
# hasta aquí tampoco funciona