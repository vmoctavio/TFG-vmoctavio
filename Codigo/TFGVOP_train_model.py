# -----------------------------------------------------------------------------------------------
# Programa: TFGVOP_train_model
# Autor: vmoctavio
# Fecha creación: 12/10/2020
# Descripción: Proceso para detectar plagas en la hoja de la vid utilizando una red sin entrenar previamente
#              Los datos origen estarán en "directory_root", dentro de una estructura de 
#              carpetas, una por cada una de las diferentes plagas a detectar.
# -----------------------------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------------------------
# Imports necesarios
# -----------------------------------------------------------------------------------------------
##
# Usar comandos del sistema operativo 
import os
#
# API,s de alto nivel para procesos de Deep Learning
import tensorflow as tf
import keras
#
# Uso de funciones del backend
import keras.backend as K
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
# Arquitecturaa de red a probar
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import Xception
#
# Convierte un modelo de Keras a diagrama y lo guarda en un archivo
#from tensorflow.keras.utils import plot_model 
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
# Librería para leer parámetros de línea de comandos
import argparse
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
v_File_Inicio_Proceso = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
v_Hora_Inicio_Proceso = datetime.datetime.now()
#
# -----------------------------------------------------------------------------------------------
# Inicio del proceso
# -----------------------------------------------------------------------------------------------
argumentos = argparse.ArgumentParser()
argumentos.add_argument("-modelo", "--modelo", type=str, default="vgg16",
	help="Nombre del modelo a entrenar")
v_argumentos = vars(argumentos.parse_args())
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
# Directorio de trabajo donde se generan los modelos de salida del proceso
directory_modelos = config.get('config','directory_modelos')
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
# Para evitar el "Gradient Value Clipping" 
CLIPVALUE = float(config.get('config','CLIPVALUE'))
# Modelos
MODELS = {
    "vgg16": VGG16,
    "vgg19": VGG19,
    "densenet121": DenseNet121,
    "densenet169": DenseNet169,
    "densenet201": DenseNet201,
    "inceptionresnetv2": InceptionResNetV2,
    "resnet50": ResNet50,
    "inceptionv3": InceptionV3,
    "xception": Xception
}
# -----------------------------------------------------------------------------------------------
# Verificamos si se trata de un modelo válido
# -----------------------------------------------------------------------------------------------
if v_argumentos["modelo"] not in MODELS.keys():
    print("[INFO]", 
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "El argumento --modelo debe tener uno de los modelos válido: vgg16, vgg19, densenet121, densenet169, densenet201, inceptionresnetv2, resnet50, inceptionv3, xception")
    quit()
#
print("[INFO]", 
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Modelo a entrenar: ",v_argumentos["modelo"])

print("Versión keras",keras.__version__)
#
# -----------------------------------------------------------------------------------------------
# Inicialización de variables
# -----------------------------------------------------------------------------------------------
#
nombre_modelo=v_argumentos["modelo"]
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
LOG_DIR=directory_log + v_argumentos["modelo"] + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
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
# Función para crear el modelo correspondiente
# -----------------------------------------------------------------------------------------------
#
def create_network():
    model = MODELS[v_argumentos["modelo"]](include_top=True,       # Incluir las 3 capas totalmente conectadas en la parte superior de la red
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
network = create_network()  
#
# -----------------------------------------------------------------------------------------------
# Inicializa parámetros para el optimizador ADAM 
# -----------------------------------------------------------------------------------------------
opt = tf.keras.optimizers.Adam(lr=INIT_LR,                 # Initial Learning Rate
                            decay=INIT_LR / EPOCHS,        # Disminución de Learning Rate 
                            clipvalue=CLIPVALUE)           # Para evitar el "Gradient Value Clipping" 
#                            momentum=0.9,                 # para probar resnet50
#                            nesterov=True)                # para probar resnet50
#
# -----------------------------------------------------------------------------------------------
# Inicializa parámetros para el optimizador ADAM 
# -----------------------------------------------------------------------------------------------
# loss_function = keras.losses.SparseCategoricalCrossentropy(from_logits=True,name="sparse_categorical_crossentropy")
#loss_function='mean_absolute_error'       # Función de pérdida
#loss_function='binary_crossentropy'       # Función de pérdida
loss_function='categorical_crossentropy'       # Función de pérdida
#loss_function='mean_squared_error'       # Función de pérdida
#loss_function='sparse_categorical_crossentropy'       # Función de pérdida
# 
# -----------------------------------------------------------------------------------------------
# Configuración / compilación del proceso de aprendizaje
# -----------------------------------------------------------------------------------------------
#
network.compile(loss=loss_function,                     # Función de pérdida
                optimizer=opt,                          # Optimizador 
                metrics=['acc', 'mse'])                 # Métricas del proceso 
#
# -----------------------------------------------------------------------------------------------
# Imprime una representación resumida del modelo
# -----------------------------------------------------------------------------------------------
#
network.summary()
#
# -----------------------------------------------------------------------------------------------
# Creamos el directorio destino de las gráficas, si no existe 
# -----------------------------------------------------------------------------------------------
try:
    os.stat(directory_log)
except:
    os.mkdir(directory_log)
#
try:
    os.stat(directory_modelos)
except:
    os.mkdir(directory_modelos)
#
# -----------------------------------------------------------------------------------------------
# Convierte un modelo de Keras a diagrama y lo guarda en un archivo
# -----------------------------------------------------------------------------------------------
#
tf.keras.utils.plot_model(model=network,
                        to_file=directory_modelos + 'Plotmodel_' + v_argumentos["modelo"] + '_' + v_File_Inicio_Proceso  + '.png',
                        show_shapes=True,
                        show_layer_names=True,
                        rankdir='TB')
#
# -----------------------------------------------------------------------------------------------
# Ejecución / entrenamiento de la red
# -----------------------------------------------------------------------------------------------
#
v_Hora_Inicio_Network = datetime.datetime.now()
#
execution_network = network.fit(x_train,                                    # Imágenes de entrenamiento
                    train_label,                                # Etiquetas entrenamiento 
                    batch_size=BATCH_SIZE,                      # Tamaño del lote 
                    epochs=EPOCHS,                              # Número de veces que se entrena la red
                    verbose=VERBOSE,                            # Barra de progreso 
                    validation_data=(x_valid, valid_label),     # Datos de validación (imágenes y etiquetas)
                    shuffle=True,                               # Reordenar los lotes al comienzo de cada epoch
                    callbacks=[tensorboard_callback])           # Configuración de Tensorboard   
v_Hora_Fin_Network = datetime.datetime.now()
#
v_Tiempo_entrenamiento = v_Hora_Fin_Network-v_Hora_Inicio_Network
#
print("[INFO]",
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Tiempo entrenamiento: ",
     v_Tiempo_entrenamiento)
#
# -----------------------------------------------------------------------------------------------
# Representación de las métricas de entrenamiento y validación
# -----------------------------------------------------------------------------------------------
#                               Training Accuracy vs Validation Accuracy
plt.figure(0)  
plt.plot(execution_network.history['acc'],'r')  
plt.plot(execution_network.history['val_acc'],'g')  
plt.xticks(np.arange(0, EPOCHS, INTERVALO))
plt.rcParams['figure.figsize'] = (8, 6)  
plt.xlabel("Num of Epochs")  
plt.ylabel("Accuracy")  
plt.title("Training Accuracy vs Validation Accuracy en " + v_argumentos["modelo"])  
plt.legend(['train','validation'])
plt.savefig(directory_modelos + 'TrainingAccuracyvsValidationAccuracy_' + v_argumentos["modelo"] + '_' + v_File_Inicio_Proceso + '.png')  
#
#                               Training Loss vs Validation Loss
plt.figure(1)  
plt.plot(execution_network.history['loss'],'r')  
plt.plot(execution_network.history['val_loss'],'g')  
plt.xticks(np.arange(0, EPOCHS, INTERVALO))
plt.rcParams['figure.figsize'] = (8, 6)  
plt.xlabel("Num of Epochs")  
plt.ylabel("Loss")  
plt.title("Training Loss vs Validation Loss en " + v_argumentos["modelo"])  
plt.legend(['train','validation'])
plt.savefig(directory_modelos + 'TrainingLossvsValidationLoss_' + v_argumentos["modelo"] + '_' + v_File_Inicio_Proceso + '.png')  
#
#                               Training mse vs Validation mse
plt.figure(2)  
plt.plot(execution_network.history['mse'],'r')  
plt.plot(execution_network.history['val_mse'],'g')  
plt.xticks(np.arange(0, EPOCHS, INTERVALO))
plt.rcParams['figure.figsize'] = (8, 6)  
plt.xlabel("Num of Epochs")  
plt.ylabel("MSE")  
plt.title("Training mse vs Validation mse en " + v_argumentos["modelo"])  
plt.legend(['train','validation'])
plt.savefig(directory_modelos + 'TrainingmsevsValidationmse_' + v_argumentos["modelo"] + '_' + v_File_Inicio_Proceso + '.png')  
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
    "Calculando la precisión del modelo " + v_argumentos["modelo"] + "... ")
#
v_Hora_Inicio_Precision = datetime.datetime.now()
#
resultado_test = network.evaluate(x_test,
                              y_test_one_hot)
#
v_Hora_Fin_Precision = datetime.datetime.now()
#
v_Tiempo_precision = v_Hora_Fin_Precision-v_Hora_Inicio_Precision
#
print("[INFO]",
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Tiempo cálculo precisión: ",
    v_Tiempo_precision)
#
print("[INFO]",      
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    f"Test Accuracy {nombre_modelo}: {resultado_test[1]*100}")
#
print("[INFO]",
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    f"Test Loss {nombre_modelo}: {resultado_test[0]}")
#
# -----------------------------------------------------------------------------------------------
# Genera las predicciones de salida para el conjunto de datos test
# -----------------------------------------------------------------------------------------------
#
predicted_classes_test = network.predict(x_test)
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
predicciones_correctas = len(correct)
#
# -----------------------------------------------------------------------------------------------
# Ejemplos de predicciones correctas
# -----------------------------------------------------------------------------------------------
# 
for i, correct in enumerate(correct[:9]):
    plt.subplot(3, 3, i+1)
    plt.imshow(x_test[correct].reshape(height,width,3), cmap='gray', interpolation='none')
    plt.title("{} / {}".format(labels[predicted_classes[correct]],labels[y_test[correct]]))
    plt.tight_layout()
#
plt.savefig(directory_modelos + 'Ejemploprediccionescorrectas_' + v_argumentos["modelo"] + '_' + v_File_Inicio_Proceso + '.png')  
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
predicciones_incorrectas = len(incorrect)

#
# -----------------------------------------------------------------------------------------------
# Ejemplos de predicciones incorrecatas
# -----------------------------------------------------------------------------------------------
#
# print("Calculado / Correcto")      
for i, incorrect in enumerate(incorrect[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[incorrect].reshape(height,width,3), cmap='gray', interpolation='none')
    plt.title("{} / {}".format(labels[predicted_classes[incorrect]],labels[y_test[incorrect]]))
    plt.tight_layout()
#
plt.savefig(directory_modelos + 'Ejemploprediccionesincorrectas_' + v_argumentos["modelo"] + '_' + v_File_Inicio_Proceso + '.png')  
plt.show()  
#
# -----------------------------------------------------------------------------------------------
# Matriz de confusión para evaluación de falsos positivos y falsos negativos
# -----------------------------------------------------------------------------------------------
#
print("[INFO]",
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Creando la matriz de confusión " + v_argumentos["modelo"] + "...")
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
plt.title('Matriz de confusión ' + v_argumentos["modelo"], pad=100, fontsize = 30, color='Black', fontstyle='italic')
plt.savefig(directory_modelos + 'Matrizdeconfusion_' + v_argumentos["modelo"] + '_' + v_File_Inicio_Proceso + '.png')  
#
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
data = {'Network': v_argumentos["modelo"],
        'Hora Inicio Proceso': v_Hora_Inicio_Proceso.strftime("%Y-%m-%d %H:%M:%S"),
        'Hora Inicio Network': v_Hora_Inicio_Network.strftime("%Y-%m-%d %H:%M:%S"),
        'Hora Fin Network': v_Hora_Fin_Network.strftime("%Y-%m-%d %H:%M:%S"),
        'Duración Network': str(v_Tiempo_entrenamiento),
        'Hora Inicio Precisión': v_Hora_Inicio_Precision.strftime("%Y-%m-%d %H:%M:%S"),
        'Hora Fin Precisión': v_Hora_Fin_Precision.strftime("%Y-%m-%d %H:%M:%S"),
        'Duración Precisión': str(v_Tiempo_precision),
        'EPOCHS': str(EPOCHS),
        'INIT_LR': str(INIT_LR),
        'BATCH_SIZE': str(BATCH_SIZE),  
        'width': str(width),
        'height': str(height),
        'depth': str(depth),
        'TEST_SIZE': str(TEST_SIZE),
        'VALID_SIZE': str(VALID_SIZE),
        'RANDOM_STATE': str(RANDOM_STATE),
        'VERBOSE': str(VERBOSE),
        'CLIPVALUE': str(CLIPVALUE),
        'ACCURACY': str(resultado_test[1]*100),
        'LOSS': str(resultado_test[0]),
        'TOTAL_IMAGES': str(len(image_list)),
        'TOTAL_LABEL': str(len(labels)),
        'loss_function': loss_function,
        'Etiquetas correctas': str(predicciones_correctas),
        'Etiquetas incorrectas': str(predicciones_incorrectas)
        }
#
TFGVOP_save_model.save_model(directory_modelos,
                            v_argumentos["modelo"] + '_' + v_File_Inicio_Proceso,
                            execution_network,
                            data)
#
print("[INFO]", 
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Fin del proceso",
    os.path.basename(__file__))