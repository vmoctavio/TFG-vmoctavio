# -----------------------------------------------------------------------------------------------
# Programa: TFGVOP_vgg19
# Autor: vmoctavio
# Fecha creación: 12/10/2019
# Descripción: Proceso para detectar plagas en la hoja de la vid utilizando una red vgg19
#              Los datos origen estarán en "directory_root", dentro de una estructura de 
#              carpetas, una por cada una de las diferentes plagas a detectar.
# -----------------------------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------------------------
# Imports necesarios
# -----------------------------------------------------------------------------------------------
#
# API de alto nivel para procesos de Deep Learning
import keras
#
# Uso de funciones del backend
from keras import backend as K
#
# Uso de funciones de manejo de imágenes
from keras.preprocessing import image
#
# Convertir una imagen en una matriz NumPy 
from keras.preprocessing.image import img_to_array
#
# Convierte un vector de clase (enteros) en una matriz de clase binaria
from keras.utils import to_categorical
#
# Optimizar para compilar un modelo
from keras.optimizers import Adam
#
# Usar comandos del sistema operativo ver si esto se borra
import os
#
# Manejo de arrays 
import numpy as np
#
# Para manipular fechas y horas
import datetime
#
# Funciones matemáticas 
import math
#
# Librería para generar gráficas
import matplotlib.pyplot as plt
#
# Entorno de trabajo para redes deep learning
import tensorflow as tf
#
# Arquitectura de red a probar
from tensorflow.keras.applications import vgg19
#
# Convierte un modelo de Keras a diagrama y lo guarda en un archivo
from tensorflow.keras.utils import plot_model 
#
# Dividir un dataset en dos
from sklearn.model_selection import train_test_split
#
# Visualizar la matriz de confusión - rendimiento de un algoritmo
from sklearn.metrics import confusion_matrix  
#
# Visualizar informe con las principales métricas de clasificación
from sklearn.metrics import classification_report  
#
# Librería para el análisis de datos
import pandas as pd  
#
# Librería para visualización de datos y gráficos
import seaborn as sn  
#
# Librería para leer archivo de configuración
import configparser
#
# Módulo para control de excepciones
import sys
#
# -----------------------------------------------------------------------------------------------
# Inicio del proceso
# -----------------------------------------------------------------------------------------------
print("[INFO]", 
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Inicio del proceso",
    os.path.basename(__file__))
#
# -----------------------------------------------------------------------------------------------
# Comprobamos si existe archivo ini y en caso contrario paramos el programa
# -----------------------------------------------------------------------------------------------
#
try:
    os.stat('TFGVOP_Config.ini')
except Exception as e: # Controla que no exista el fichero
    print("[ERROR]",
          datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
          "No existe el fichero de configuración",
          sys.exc_info()[0],
          sys.exc_info()[1],
          sys.exc_info()[2])    
    sys.exit()
#
# -----------------------------------------------------------------------------------------------
# Leemos archivo ini e inicializamos variables
# -----------------------------------------------------------------------------------------------
#
print("[INFO]", 
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Leyendo archivo de configuración...")
#
config = configparser.ConfigParser()
config.read('TFGVOP_Config.ini')
#
# Directorio de trabajo donde está el dataset
directory_root = config.get('config','directory_root')
# Directorio de trabajo donde se generan los logs de salida del proceso
directory_log = config.get('config','directory_log')
# Número de veces que se entrena la red
EPOCHS = int(config.get('config','EPOCHS'))
# Initial Learning Rate
INIT_LR = float(config.get('config','INIT_LR'))
# Tamaño del lote
BATCH_SIZE = int(config.get('config','BATCH_SIZE'))
# Tamaño de la imagen: ancho
width = int(config.get('config','width'))
# Tamaño de la imagen: alto
height = int(config.get('config','height'))
# Número de canales de la imagen
depth = int(config.get('config','depth'))
# Porcentaje división dataset en train y test
TEST_SIZE = float(config.get('config','TEST_SIZE'))
# Porcentaje división dataset en train y validation (a partir del subconjunto anterior)
VALID_SIZE = float(config.get('config','VALID_SIZE'))
# Division del dataset no aleatorio
RANDOM_STATE = int(config.get('config','RANDOM_STATE'))
# Training progress
VERBOSE = int(config.get('config','VERBOSE'))
#
# -----------------------------------------------------------------------------------------------
# Inicialización de variables
# -----------------------------------------------------------------------------------------------
#
image_list=[]       # array de array's resultado de convertir las imágenes del directorio
image_labels=[]     # array de array's de etiquetas
labels=[]           # array con las distintas etiquetas del dataset
#
# Intervalo para las coordenadas de la gráfica
INTERVALO = math.ceil(EPOCHS/10)
#
# -----------------------------------------------------------------------------------------------
# Llamada a la función load_dataset_process para cargar arrays de imágenes y etiquetas
# -----------------------------------------------------------------------------------------------
#
# Función para cargar un Dataset a partir de un determinadao directorio
import TFGVOP_load_dataset
#
print("[INFO]", 
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Llamada al proceso TFGVOP_load_dataset ")
#
(image_labels,image_list,labels) = TFGVOP_load_dataset.load_dataset_process(directory_root)
#
print("[INFO]", 
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Imágenes cargadas. Número total de imágenes para procesar:  ",
    len(image_list),
    ". Número total de clases diferentes:",
    len(labels))
#
# -----------------------------------------------------------------------------------------------
# Calculamos el número de clases diferentes
# -----------------------------------------------------------------------------------------------
#
n_classes = len(labels)
#
# -----------------------------------------------------------------------------------------------
# Convertir la lista de imágenes y etiquetas en arrays Numpy
# -----------------------------------------------------------------------------------------------
#
np_image_list = np.array(image_list, dtype=np.uint8)
y = np.array(image_labels)
#
# -----------------------------------------------------------------------------------------------
# División de los datos (imágenes y etiquetas) 
# en archivos para entrenamiento (train) y pruebas (test)
# -----------------------------------------------------------------------------------------------
#
print("[INFO]", 
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Dividiendo datos en train y test...")
#
x_train, x_test, y_train, y_test = train_test_split(np_image_list,          # array de imágenes
                                                    y,                      # array de etiquetas
                                                    test_size=TEST_SIZE,    # % para el archivo de test 
                                                    random_state = RANDOM_STATE)
#
print("[INFO]", 
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Training data shape (x_train): ",
    x_train.shape)
print("[INFO]", 
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Training data shape (y_train): ",
    y_train.shape)
#
print("[INFO]", 
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Testing data shape (x_test): ",
    x_test.shape)
print("[INFO]", 
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Testing data shape (y_test): ",
    y_test.shape)
#
# -----------------------------------------------------------------------------------------------
# Cambiamos los tipos y escalamos los pixeles en el ranto [0,1]
# -----------------------------------------------------------------------------------------------
#
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.
x_test = x_test / 255.
#
# -----------------------------------------------------------------------------------------------
# Convierte un vector de clase (enteros) en una matriz de clase binaria
# -----------------------------------------------------------------------------------------------
# 
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)
#
# -----------------------------------------------------------------------------------------------
# División de los datos (imágenes y etiquetas) 
# en archivos para entrenamiento (train) y validación (valid)
# a partir de los archivos de entrenamiento que se generaron en la primera división.
# -----------------------------------------------------------------------------------------------
#
print("[INFO]",
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Dividiendo datos en train y valid...")
#
x_train, x_valid, train_label, valid_label = train_test_split(x_train,          # array de imágenes 
                                                    y_train_one_hot,            # array de etiquetas
                                                    test_size=VALID_SIZE,       # % para el archivo de validación  
                                                    random_state = RANDOM_STATE)
#
print("[INFO]", 
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Training data shape (x_train): ",
    x_train.shape)
print("[INFO]", 
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Training data shape (train_label): ",
    train_label.shape)
#
print("[INFO]", 
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Validating data shape (x_valid): ",
    x_valid.shape)
print("[INFO]", 
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Validating data shape (valid_label): ",
    valid_label.shape)
#
# -----------------------------------------------------------------------------------------------
# Inicialización archivo de registro (LOG_DIR) y Tensorboard 
# -----------------------------------------------------------------------------------------------
#
LOG_DIR=directory_log + 'vgg19_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# Frecuencia (en epochs) a la que se calculan los histogramas de activación para las capas del modelo.
v_histogram_freq=1  
#
print("[INFO]",
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Generado log en... ",
    LOG_DIR)
#
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR,      # Directorio de registro
                                                    histogram_freq=v_histogram_freq,       
                                                    write_graph=True,       # Si visualizar el gráfico
                                                    write_images=True,      # Si visualizar imágenes
                                                    update_freq='epoch',    # Las métricas se generan por cada epoch
                                                    profile_batch=2)        # Perfilar el segundo lote
#
# -----------------------------------------------------------------------------------------------
# Inicializa input_shape en función del formato de la imagen
# -----------------------------------------------------------------------------------------------
#
inputShape = (height, width, depth)
chanDim = -1
if K.image_data_format() == "channels_first":
    inputShape = (depth, height, width)
    chanDim = 1
#
# -----------------------------------------------------------------------------------------------
# Función para crear el modelo vgg19
# -----------------------------------------------------------------------------------------------
#
def create_vgg19():
    model = vgg19.VGG19(include_top=True,       # Incluir las 3 capas totalmente conectadas en la parte superior de la red
                        weights=None,           # Inicialización random (no partir de pre-entrenadas de imagenet)
                        input_tensor=None,      # No usar tensor como entrada de imágenes
                        input_shape=inputShape, # Resolución de las imágenes de entrada
                        pooling=None,           # Modo agrupación de características
                        classes=n_classes)      # Número de clases
    return model
#
# -----------------------------------------------------------------------------------------------
# Llamada a la función para crear el modelo
# -----------------------------------------------------------------------------------------------
#
vgg19_model = create_vgg19()  
#
# -----------------------------------------------------------------------------------------------
# Inicializa parámetros para el optimizador ADAM 
# -----------------------------------------------------------------------------------------------
#
opt = tf.keras.optimizers.Adam(lr=INIT_LR,                  # Initial Learning Rate
                               decay=INIT_LR / EPOCHS)      # Disminución de Learning Rate 
#
# -----------------------------------------------------------------------------------------------
# Configuración / compilación del proceso de aprendizaje
# -----------------------------------------------------------------------------------------------
#
vgg19_model.compile(loss='categorical_crossentropy',    # Función de pérdida
                    optimizer=opt,                      # Optimizador 
                    metrics=['acc', 'mse'])             # Métricas del proceso 
#
# -----------------------------------------------------------------------------------------------
# Imprime una representación resumida del modelo
# -----------------------------------------------------------------------------------------------
#
vgg19_model.summary()
#
# -----------------------------------------------------------------------------------------------
# Creamos el directorio destino de las gráficas, si no existe 
# -----------------------------------------------------------------------------------------------
try:
    os.stat(directory_log + 'modelos')
except:
    os.mkdir(directory_log + 'modelos')
#
# -----------------------------------------------------------------------------------------------
# Convierte un modelo de Keras a diagrama y lo guarda en un archivo
# -----------------------------------------------------------------------------------------------
#
tf.keras.utils.plot_model(model=vgg19_model,
                       to_file=directory_log + 'modelos/' + 'Plotmodel_vgg19.png',
                       show_shapes=True,
                       show_layer_names=True,
                       rankdir='TB',
                       expand_nested=True,
                       dpi=96)
#
# -----------------------------------------------------------------------------------------------
# Ejecución / entrenamiento de la red
# -----------------------------------------------------------------------------------------------
#
vgg19 = vgg19_model.fit(x_train,                                    # Imágenes de entrenamiento
                        train_label,                                # Etiquetas entrenamiento 
                        batch_size=BATCH_SIZE,                      # Tamaño del lote 
                        epochs=EPOCHS,                              # Número de veces que se entrena la red
                        verbose=VERBOSE,                            # Barra de progreso 
                        validation_data=(x_valid, valid_label),     # Datos de validación (imágenes y etiquetas)
                        shuffle=True,                               # Reordenar los lotes al comienzo de cada epoch
                        callbacks=[tensorboard_callback])           # Configuración de Tensorboard   
#
# -----------------------------------------------------------------------------------------------
# Representación de las métricas de entrenamiento y validación
# -----------------------------------------------------------------------------------------------
#                               Training Accuracy vs Validation Accuracy
plt.figure(0)  
plt.plot(vgg19.history['acc'],'r')  
plt.plot(vgg19.history['val_acc'],'g')  
plt.xticks(np.arange(0, EPOCHS, INTERVALO))
plt.rcParams['figure.figsize'] = (8, 6)  
plt.xlabel("Num of Epochs")  
plt.ylabel("Accuracy")  
plt.title("Training Accuracy vs Validation Accuracy en vgg19")  
plt.legend(['train','validation'])
plt.savefig(directory_log + 'modelos/' + 'TrainingAccuracyvsValidationAccuracy_en_vgg19.png')  

#                               Training Loss vs Validation Loss
plt.figure(1)  
plt.plot(vgg19.history['loss'],'r')  
plt.plot(vgg19.history['val_loss'],'g')  
plt.xticks(np.arange(0, EPOCHS, INTERVALO))
plt.rcParams['figure.figsize'] = (8, 6)  
plt.xlabel("Num of Epochs")  
plt.ylabel("Loss")  
plt.title("Training Loss vs Validation Loss en vgg19")  
plt.legend(['train','validation'])
plt.savefig(directory_log + 'modelos/' + 'TrainingLossvsValidationLoss_en_vgg19.png')  

#                               Training mse vs Validation mse
plt.figure(2)  
plt.plot(vgg19.history['mse'],'r')  
plt.plot(vgg19.history['val_mse'],'g')  
plt.xticks(np.arange(0, EPOCHS, INTERVALO))
plt.rcParams['figure.figsize'] = (8, 6)  
plt.xlabel("Num of Epochs")  
plt.ylabel("MSE")  
plt.title("Training mse vs Validation mse en vgg19")  
plt.legend(['train','validation'])
plt.savefig(directory_log + 'modelos/' + 'TrainingmsevsValidationmse_en_vgg19.png')  
#
plt.show()
#
# -----------------------------------------------------------------------------------------------
# Calcular la precisión del modelo con el conjunto de datos de test 
# los cuales no han participado en el proceso de entrenamiento y validación
# -----------------------------------------------------------------------------------------------
#
print("[INFO]",
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Calculando la precisión del modelo vgg19... ")
#
resultado_test = vgg19_model.evaluate(x_test,
                              y_test_one_hot)
#
print("[INFO]",
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    f"Test Accuracy vgg19: {resultado_test[1]*100}")
#
print("[INFO]",
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    f"Test Loss vgg19: {resultado_test[0]*100}")  # no sé si el loss se multiplica o no
#
# -----------------------------------------------------------------------------------------------
# Genera las predicciones de salida para el conjunto de datos test
# -----------------------------------------------------------------------------------------------
#
predicted_classes_test = vgg19_model.predict(x_test)
#
predicted_classes=[]
for predicted_image in predicted_classes_test:
    predicted_classes.append(predicted_image.tolist().index(max(predicted_image)))
#
predicted_classes=np.array(predicted_classes)
#
# -----------------------------------------------------------------------------------------------
# Convertir array de etiquetas de test
# -----------------------------------------------------------------------------------------------
#
y_test_aux = np.array(y_test)
#
predicted_classes.shape, y_test.shape
#
# -----------------------------------------------------------------------------------------------
# Calcular predicciones correctas
# -----------------------------------------------------------------------------------------------
#
correct = np.where(predicted_classes==y_test)[0]
#
print("[INFO]",
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Se han encontrado %d etiquetas correctas" % len(correct))
#
# -----------------------------------------------------------------------------------------------
# Ejemplos de predicciones correctas
# -----------------------------------------------------------------------------------------------
#
for i, correct in enumerate(correct[0:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[correct].reshape(256,256,3), cmap='gray', interpolation='none')
    plt.title("{} / {}".format(labels[predicted_classes[correct]],labels[y_test[correct]]))
    plt.tight_layout()
#
plt.savefig(directory_log + 'modelos/' + 'Ejemploprediccionescorrectas_en_vgg19.png')  
plt.show()  
#
# -----------------------------------------------------------------------------------------------
# Calcular predicciones incorrectas
# -----------------------------------------------------------------------------------------------
#
incorrect = np.where(predicted_classes!=y_test)[0]
#
print("[INFO]",
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Se han encontrado %d etiquetas incorrectas" % len(incorrect))
#
# -----------------------------------------------------------------------------------------------
# Ejemplos de predicciones incorrecatas
# -----------------------------------------------------------------------------------------------
#
# print("Calculado / Correcto")
for i, incorrect in enumerate(incorrect[0:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[incorrect].reshape(256,256,3), cmap='gray', interpolation='none')
    plt.title("{} / {}".format(labels[predicted_classes[incorrect]],labels[y_test[incorrect]]))
    plt.tight_layout()
#
plt.savefig(directory_log + 'modelos/' + 'Ejemploprediccionesincorrectas_en_vgg19.png')  
plt.show()  
#
# -----------------------------------------------------------------------------------------------
# Matriz de confusión para evaluación de falsos positivos y falsos negativos
# -----------------------------------------------------------------------------------------------
#
print("[INFO]",
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Creando la matriz de confusión vgg19...")
#
# -----------------------------------------------------------------------------------------------
# Creamos la matriz de confusión
# -----------------------------------------------------------------------------------------------
#
predicted_classes_test_confusion = np.argmax(predicted_classes_test, axis=1)  
#
predicted_classes_test_confusion_cm = confusion_matrix(np.argmax(y_test_one_hot, axis=1),
                                                       predicted_classes_test_confusion)
#
print("[INFO]",
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Visualizando la matriz de confusión...")
#
# -----------------------------------------------------------------------------------------------
# Visualizamos la matriz de confusión. Aquí se visualiza el desempeño del algoritmo.
# Cada columna de la matriz representa el número de predicciones de cada etiqueta y 
# las filas la instancia de la etiqueta real.
# -----------------------------------------------------------------------------------------------
#
predicted_classes_test_confusion_cm_df = pd.DataFrame(predicted_classes_test_confusion_cm,
                                                      columns=labels,
                                                      index=labels)
#
plt.figure(figsize = (20,14))  
#
sn.set(font_scale=2)                # Tamaño de fuente mapa calor
sn.heatmap(predicted_classes_test_confusion_cm_df,
           annot=True,              # Escribe el número de coincidencias en cada celda
           linewidth=0.5,           # Ancho del borde de celdas de la tabla
           cmap="YlOrRd",           # Mapa de color amarillo - naranja - rojo
           square=True,             # Forzar tamaño celdas
           annot_kws={"size": 20})  # Tamaño de fuente barra
plt.title('Matriz de confusión vgg19', pad=100, fontsize = 30, color='Black', fontstyle='italic')
plt.savefig(directory_log + 'modelos/' + 'Matrizdeconfusion_en_vgg19.png')  

plt.show() 
#
# -----------------------------------------------------------------------------------------------
# Visualizamos informe resumen de clasificación
# -----------------------------------------------------------------------------------------------
#
cm_report = classification_report(np.argmax(y_test_one_hot, axis=1),
                                predicted_classes_test_confusion,
                                target_names = labels)  
#
print("[INFO]", 
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Informe de clasificación")
print(cm_report)
#
# -----------------------------------------------------------------------------------------------
# Guardamos el modelo para después poder compararlos
# -----------------------------------------------------------------------------------------------
#
print("[INFO]", 
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Llamada al proceso TFGVOP_save_model ")
#
import TFGVOP_save_model
#
TFGVOP_save_model.save_model(directory_log,'vgg19',vgg19)

print("[INFO]", 
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Fin del proceso",
    os.path.basename(__file__))