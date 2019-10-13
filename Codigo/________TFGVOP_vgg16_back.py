# -----------------------------------------------------------------------------------------------
# Programa: TFGVOP_vgg16
# Autor: vmoctavio
# Fecha creación: 28/09/2019
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

import TFGVOP_load_dataset


# Para qué sirve el
import keras
#
# Para qué sirve el
from keras import backend as K
#
# Para qué sirve el   ver si algo de esto sobra......
from keras.preprocessing import image
#
# Convertir una imagen en una matriz NumPy 
from keras.preprocessing.image import img_to_array
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
from keras.layers import Activation
#
# Para qué sirve el
from keras.layers import Dense
#
# Para qué sirve el
from keras.layers import Dropout
#
# Para qué sirve el
from keras.layers import Flatten
#
# Para qué sirve el
from keras.layers.normalization import BatchNormalization
#
# Para qué sirve el
from keras.models import Sequential
#
# Usar comandos del sistema operativo ver si esto se borra
#import os
#
# ver si esto puede sobrar ver si esto se borra
#import os.path
#
# Devuelve una lista que contiene el nombre de las entradas en el directorio de la ruta del parámetro.
from os import listdir
#
# Manejo de arrays 
import numpy as np
#
# Para manipular fechas y horas
import datetime
#
#Para qué sirve el cv2
import cv2
#
# Para qué sirve el matplotlib.pyplot
import matplotlib.pyplot as plt
#
# Para qué sirve el pickle ver si puede sobrar donde se usa......
import pickle
#
# Para qué sirve el
from sklearn.model_selection import train_test_split
#
# Para qué sirve el
from sklearn.preprocessing import LabelBinarizer
#
# Para qué sirve el
from sklearn.preprocessing import MultiLabelBinarizer
#
# Para qué sirve el ver si esto se borra
#import shutil
#
# Para qué sirve el
import tensorflow as tf
#
# Para qué sirve el
from tensorflow.keras.applications import vgg16
#

from sklearn.metrics import confusion_matrix, classification_report  

import pandas as pd  
import seaborn as sn  


#
# -----------------------------------------------------------------------------------------------
# Inicio del proceso
# -----------------------------------------------------------------------------------------------
print("[INFO]", 
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Inicio del proceso TFGVOP_vgg16 ")
#
# -----------------------------------------------------------------------------------------------
# Inicialización de variables
# -----------------------------------------------------------------------------------------------
#
# Directorio de trabajo donde está el dataset
directory_root = '/Users/vmoctavio/Downloads/keras-tutorial-dataaugmentation/Dataset_vid_prueba_dest/'
# Directorio de trabajo donde se generan los logs de salida del proceso
directory_log = '/Users/vmoctavio/Downloads/keras-tutorial/logs/pruebavidcon/borrar/'
#
# Número de veces que se procesará el dataset de entrada
#EPOCHS = 150
EPOCHS = 10
#EPOCHS = 135
# para qué es esto
#INIT_LR = 1e-3   esto es lo que vi en otro sitio pero es 0.001
# para qué es esto
INIT_LR = 0.00001
# para qué es esto
#BATCH_SIZE = 128
BATCH_SIZE = 32
#BS = 32 esto creo que se podrá borrar
# para qué es esto
image_size = 0
# para qué es esto
default_image_size = tuple((256, 256))
#default_image_size = tuple((48, 48))
# para qué es esto
width=256 #antes 256
# para qué es esto
height=256
# para qué es esto
depth=3
# para qué es esto
TEST_SIZE = 0.2
# para qué es esto
RANDOM_STATE = 42
#
image_list=[]       # array de array's resultado de convertir las imágenes del directorio
image_labels=[]     # array de array's de etiquetas
labels=[]           # array con las distintas etiquetas del dataset
#
# -----------------------------------------------------------------------------------------------
# Llamada a la función load_dataset_process para cargar arrays de imágenes y etiquetas
# -----------------------------------------------------------------------------------------------
#
print("[INFO]", 
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Llamada al proceso TFGVOP_load_dataset ")
#
(image_labels,image_list,labels) = TFGVOP_load_dataset.load_dataset_process(directory_root)
#
print("[INFO]", 
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Ver que poner aquí que es cuando volvemos ")
#
print("[INFO]", 
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Número total de imágenes para procesar: ",
    len(image_list))
#
# -----------------------------------------------------------------------------------------------
# Calculamos el número de clases diferentes
# -----------------------------------------------------------------------------------------------
#
n_classes = len(image_labels)


#
# -----------------------------------------------------------------------------------------------
# Proceso para leer las carpetas del directorio del parámetro
# y genera un array de imágenes y etiquetas con el contenido de las mismas
# 
# -----------------------------------------------------------------------------------------------
#

#
# -----------------------------------------------------------------------------------------------
# Obtener el tamño de la imagen
# -----------------------------------------------------------------------------------------------
#
#image_size = len(image_list) ver si esto se puede borar
#

# -----------------------------------------------------------------------------------------------
# xxxxxx
# -----------------------------------------------------------------------------------------------
#
np_image_list = np.array(image_list, dtype=np.float16) / 225.0   # no sé para qué es lo del / 225.0
#
# -----------------------------------------------------------------------------------------------
# xxxxxx
# -----------------------------------------------------------------------------------------------
#
print("[INFO]", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Dividiendo datos en train y test...")
#
x_train, x_test, y_train, y_test = train_test_split(np_image_list, 
                                                    image_labels, 
                                                    test_size=TEST_SIZE, 
                                                    random_state = RANDOM_STATE)
#
# -----------------------------------------------------------------------------------------------
# xxxxxx
# -----------------------------------------------------------------------------------------------
#
LOG_DIR=directory_log + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#
print("[INFO]", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Generado log en... ",LOG_DIR)
#
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR,
                                                    histogram_freq=1, 
                                                    write_graph=True,
                                                    write_images=True,
                                                    update_freq='epoch',
                                                    profile_batch=2)

#input_shape = (224, 224, 3)

#model = tf.keras.Sequential()
#
# -----------------------------------------------------------------------------------------------
# xxxxxx
# -----------------------------------------------------------------------------------------------
#
inputShape = (height, width, depth)
chanDim = -1
if K.image_data_format() == "channels_first":
    inputShape = (depth, height, width)
    chanDim = 1
#
# -----------------------------------------------------------------------------------------------
# xxxxxx
# -----------------------------------------------------------------------------------------------
#
def create_vgg16():
    model = vgg16.VGG16(include_top=True, weights=None, input_tensor=None, input_shape=inputShape, pooling=None, classes=n_classes)
    return model
#
# -----------------------------------------------------------------------------------------------
# xxxxxx
# -----------------------------------------------------------------------------------------------
#
vgg16_model = create_vgg16()  
#
# -----------------------------------------------------------------------------------------------
# xxxxxx
# -----------------------------------------------------------------------------------------------
#
opt = tf.keras.optimizers.Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
#
# -----------------------------------------------------------------------------------------------
# xxxxxx
# -----------------------------------------------------------------------------------------------
#
vgg16_model.compile(loss='categorical_crossentropy', 
                    optimizer=opt, 
                    metrics=['acc', 'mse', 'mae', 'mape', 'cosine_proximity','binary_accuracy','categorical_accuracy',])  
#
# -----------------------------------------------------------------------------------------------
# xxxxxx
# -----------------------------------------------------------------------------------------------
#
vgg16_model.summary()
#
# -----------------------------------------------------------------------------------------------
# xxxxxx
# -----------------------------------------------------------------------------------------------
#
vgg16 = vgg16_model.fit(x=x_train, 
                        y=y_train, 
                        batch_size=BATCH_SIZE, 
                        epochs=EPOCHS, 
                        verbose=1, 
                        validation_data=(x_test, y_test), 
                        shuffle=True, 
                        callbacks=[tensorboard_callback])  
#
# -----------------------------------------------------------------------------------------------
# xxxxxxx esto es posible que se tenga que borrar es para ver qué sale
# -----------------------------------------------------------------------------------------------
#
# XXXXXXXX
print("vgg16.history.keys(): ",vgg16.history.keys())
#


print("Training data shape: ", x_train.shape, y_train.shape)
print("Testing data shape: ", x_test.shape, y_test.shape)


# -----------------------------------------------------------------------------------------------
# Model Accuracy
# -----------------------------------------------------------------------------------------------
#
print("[INFO] Calculating model accuracy")
scores = vgg16_model.evaluate(x_test, y_test)
print(f"Test Accuracy: {scores[1]*100}")
print(f"Test Loss: {scores[0]*100}")  # no sé si el loss se multiplica o no

print(f"el 0: {scores[0]}")  # esto es el loss sin multiplizar
#
# ver qué es el cero qu eel número no me cuadra y lo que imprime antes de loss: tampoco sé qué dato es para interpretarlo
print("ACCURACY :")
print(f"Training accuracy : {vgg16.history['acc'][-1]}")    # por qué -1 en lugar de 1 como arriba
print(f"Validation accuracy : {vgg16.history['val_acc'][-1]}")
    
print("\nLOSS :")
print(f"Training categorical crossentropy loss : {vgg16.history['loss'][-1]}")
print(f"Validation categorical crossentropy loss : {vgg16.history['val_loss'][-1]}")



# ver qué es el cero qu eel número no me cuadra y lo que imprime antes de loss: tampoco sé qué dato es para interpretarlo
print("ACCURACY :")
print(f"Training accuracy : {vgg16.history['acc'][-1]}")
print(f"Validation accuracy : {vgg16.history['val_acc'][-1]}")

print("ACCURACY 1:")
print(f"Training accuracy : {vgg16.history['acc'][1]}")
print(f"Validation accuracy : {vgg16.history['val_acc'][1]}")

print("ACCURACY sin índice:")
print(f"Training accuracy : {vgg16.history['acc']}")
print(f"Validation accuracy : {vgg16.history['val_acc']}")


print("\nLOSS :")
print(f"Training categorical crossentropy loss : {vgg16.history['loss'][-1]}")
print(f"Validation categorical crossentropy loss : {vgg16.history['val_loss'][-1]}")


print("\nLOSS 1 :")
print(f"Training categorical crossentropy loss : {vgg16.history['loss'][1]}")
print(f"Validation categorical crossentropy loss : {vgg16.history['val_loss'][1]}")

print("\nLOSS sin índice :")
print(f"Training categorical crossentropy loss : {vgg16.history['loss']}")
print(f"Validation categorical crossentropy loss : {vgg16.history['val_loss']}")




#
# -----------------------------------------------------------------------------------------------
# Model Accuracy
# -----------------------------------------------------------------------------------------------
#




plt.figure(0)  
plt.plot(vgg16.history['acc'],'r')  
plt.plot(vgg16.history['val_acc'],'g')  
plt.xticks(np.arange(0, EPOCHS, 1.0))  
plt.rcParams['figure.figsize'] = (8, 6)  
plt.xlabel("Num of Epochs")  
plt.ylabel("Accuracy")  
plt.title("Training Accuracy vs Validation Accuracy")  
plt.legend(['train','validation'])

plt.figure(1)  
plt.plot(vgg16.history['loss'],'r')  
plt.plot(vgg16.history['val_loss'],'g')  
plt.xticks(np.arange(0, EPOCHS, 1.0))  
plt.rcParams['figure.figsize'] = (8, 6)  
plt.xlabel("Num of Epochs")  
plt.ylabel("Loss")  
plt.title("Training Loss vs Validation Loss")  
plt.legend(['train','validation'])

plt.figure(2)  
plt.plot(vgg16.history['mse'],'r')  
plt.plot(vgg16.history['val_mse'],'g')  
plt.xticks(np.arange(0, EPOCHS, 1.0))  
plt.rcParams['figure.figsize'] = (8, 6)  
plt.xlabel("Num of Epochs")  
plt.ylabel("MSE")  
plt.title("Training mse vs Validation mse")  
plt.legend(['train','validation'])

plt.figure(3)  
plt.plot(vgg16.history['mae'],'r')  
plt.plot(vgg16.history['val_mae'],'g')  
plt.xticks(np.arange(0, EPOCHS, 1.0))  
plt.rcParams['figure.figsize'] = (8, 6)  
plt.xlabel("Num of Epochs")  
plt.ylabel("MAE")  
plt.title("Training mae vs Validation mae")  
plt.legend(['train','validation'])

plt.show()  


predicted_classes_test = vgg16_model.predict(x_test)

predicted_classes=[]
for predicted_image in predicted_classes_test:
    predicted_classes.append(vgg16_model.tolist().index(max(predicted_image)))
predicted_classes=np.array(predicted_classes)

predicted_classes.shape, y_test.shape


correct = np.where(predicted_classes==y_test)[0]
print("Found %d correct labels" % len(correct))
for i, correct in enumerate(correct[0:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[correct].reshape(256,256,3), cmap='gray', interpolation='none')
    plt.title("{}, {}".format(labels[predicted_classes[correct]],labels[y_test[correct]]))

    plt.tight_layout()

incorrect = np.where(predicted_classes!=y_test)[0]
print("Found %d incorrect labels" % len(incorrect))
for i, incorrect in enumerate(incorrect[0:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[incorrect].reshape(256,256,3), cmap='gray', interpolation='none')
    plt.title("{}, {}".format(labels[predicted_classes[incorrect]],labels[y_test[incorrect]]))
    plt.tight_layout()

target_names = ["Class {}".format(i) for i in range(n_classes)]
print(classification_report(y_test, predicted_classes, target_names=target_names))

#Creamos la matriz de confusión
vgg16_pred = vgg16_model.predict(x_test, batch_size=32, verbose=1)  
vgg16_predicted = np.argmax(vgg16_pred, axis=1)  

vgg16_cm = confusion_matrix(np.argmax(y_test, axis=1), vgg16_predicted)

# Visualiamos la matriz de confusión
vgg16_df_cm = pd.DataFrame(vgg16_cm, range(4), range(4))  
plt.figure(figsize = (20,14))  
sn.set(font_scale=1.4) #for label size  
sn.heatmap(vgg16_df_cm, annot=True, annot_kws={"size": 12}) # font size  
plt.show()  

print("antes del report")

vgg16_report = classification_report(np.argmax(y_test, axis=1), vgg16_predicted)  
print(vgg16_report)




def plot_lr(history):
lr=1e-4
fig, ax = plt.subplots(figsize=(7, 5))

# Plot learning rate
ax.plot(vgg16.history['lr'])
ax.set_title('Learning rate evolution')
ax.set_ylabel('Learning rate value')
ax.set_xlabel('Epochs')
ax.legend(['Train'], loc='upper right')

plot_lr(vgg16)  

print("[INFO]", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Fin del proceso TFGVOP_vgg16 ")

