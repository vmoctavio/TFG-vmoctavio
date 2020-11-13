# -----------------------------------------------------------------------------------------------
# Programa: TFGVOP_Comparativa_graficas
# Autor: vmoctavio
# Fecha creación: 13/10/2019
# Descripción: Proceso que muestra gráficas comparativas con las estadísticas de las diferentes
#              redes utilizadas en el proyecto.
#              Los datos origen estarán en "directory_modelos"
# -----------------------------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------------------------
# Imports necesarios
# -----------------------------------------------------------------------------------------------
#
# API de alto nivel para procesos de Deep Learning
import keras
#
# Usar comandos del sistema operativo ver si esto se borra
import os
#
# Módulo para control de excepciones
import sys
#
# Para manipular fechas y horas
import datetime
#
# Librería para serializar objetos
import pickle
#
# Librería para generar gráficas
import matplotlib.pyplot as plt
# 
# Manejo de arrays 
import numpy as np
#
# Funciones matemáticas 
import math
#
# Librería para leer archivo de configuración
import configparser
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
directory_modelos = config.get('config','directory_modelos')
#
# -----------------------------------------------------------------------------------------------
# Proceso principal: lee las estadísticas de los diferentes modelos del directorio "directory_modelos"
# -----------------------------------------------------------------------------------------------
#
print("[INFO]", 
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Recuperando vgg16.stats...")
#
with open(directory_modelos + 'vgg16.stats', 'rb') as file_modelo:  
  vgg16_history = pickle.load(file_modelo)
#
#
print("[INFO]", 
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Recuperando vgg19.stats...")
#
with open(directory_modelos + 'vgg19.stats', 'rb') as file_modelo:  
  vgg19_history = pickle.load(file_modelo)
#
#
print("[INFO]", 
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Recuperando ResNet152V2.stats...")
#
with open(directory_modelos + 'ResNet152V2.stats', 'rb') as file_modelo:  
  ResNet152V2_history = pickle.load(file_modelo)
#
#
print("[INFO]", 
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Recuperando DenseNet201.stats...")
#
with open(directory_modelos + 'DenseNet201.stats', 'rb') as file_modelo:  
  DenseNet201_history = pickle.load(file_modelo)
#
#
print("[INFO]", 
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Recuperando InceptionResNetV2.stats...")
#
with open(directory_modelos + 'InceptionResNetV2.stats', 'rb') as file_modelo:  
  InceptionResNetV2_history = pickle.load(file_modelo)
#
#
print("[INFO]", 
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Recuperando InceptionResNetV2.stats...")
#
with open(directory_modelos + 'CNN_1.stats', 'rb') as file_modelo:  
  CNN_1_history = pickle.load(file_modelo)
#
# -----------------------------------------------------------------------------------------------
# Calculamos número EPOCHS para gráficos
# -----------------------------------------------------------------------------------------------
# 
EPOCHS = len(vgg16_history['acc'])
#
# Intervalo para las coordenadas de la gráfica
INTERVALO = math.ceil(EPOCHS/10)
#
# -----------------------------------------------------------------------------------------------
# Representación de las métricas comparativas
# -----------------------------------------------------------------------------------------------
#                               
plt.figure(0)  
plt.plot(vgg16_history['acc'],'r')
plt.plot(vgg19_history['acc'],'g')  
plt.plot(ResNet152V2_history['acc'],'b')  
plt.plot(DenseNet201_history['acc'],'y')  
plt.plot(InceptionResNetV2_history['acc'],'m')
plt.plot(CNN_1_history['acc'],'c')
plt.xticks(np.arange(0, EPOCHS, INTERVALO))
plt.rcParams['figure.figsize'] = (8, 6)  
plt.xlabel("Num of Epochs")  
plt.ylabel("Accuracy")  
plt.title("Comparativa Accuracy en diferentes modelos")  
plt.legend(['vgg16','vgg19','ResNet152V2','DenseNet201','InceptionResNetV2','CNN_1'])
plt.savefig(directory_modelos + 'Comparativa_Accuracy.png')  
#
plt.figure(1)  
plt.plot(vgg16_history['loss'],'r')
plt.plot(vgg19_history['loss'],'g')  
plt.plot(ResNet152V2_history['loss'],'b')  
plt.plot(DenseNet201_history['loss'],'y')
plt.plot(InceptionResNetV2_history['loss'],'m')  
plt.plot(CNN_1_history['loss'],'c')  
plt.xticks(np.arange(0, EPOCHS, INTERVALO))
plt.rcParams['figure.figsize'] = (8, 6)  
plt.xlabel("Num of Epochs")  
plt.ylabel("Loss")  
plt.title("Comparativa loss en diferentes modelos")  
plt.legend(['vgg16','vgg19','ResNet152V2','DenseNet201','InceptionResNetV2','CNN_1'])
plt.savefig(directory_modelos + 'Comparativa_loss.png')  
#
plt.figure(2)  
plt.plot(vgg16_history['mse'],'r')
plt.plot(vgg19_history['mse'],'g')  
plt.plot(ResNet152V2_history['mse'],'b')  
plt.plot(DenseNet201_history['mse'],'y')  
plt.plot(InceptionResNetV2_history['mse'],'m')  
plt.plot(CNN_1_history['mse'],'c')  
plt.xticks(np.arange(0, EPOCHS, INTERVALO))
plt.rcParams['figure.figsize'] = (8, 6)  
plt.xlabel("Num of Epochs")  
plt.ylabel("MSE")  
plt.title("Comparativa mse en diferentes modelos")  
plt.legend(['vgg16','vgg19','ResNet152V2','DenseNet201','InceptionResNetV2','CNN_1'])
plt.savefig(directory_modelos + 'Comparativa_mse.png')  
#                           
plt.figure(3)  
plt.plot(vgg16_history['val_acc'],'r')
plt.plot(vgg19_history['val_acc'],'g')  
plt.plot(ResNet152V2_history['val_acc'],'b')  
plt.plot(DenseNet201_history['val_acc'],'y')  
plt.plot(InceptionResNetV2_history['val_acc'],'m')  
plt.plot(CNN_1_history['val_acc'],'c')  
plt.xticks(np.arange(0, EPOCHS, INTERVALO))
plt.rcParams['figure.figsize'] = (8, 6)  
plt.xlabel("Num of Epochs")  
plt.ylabel("Accuracy")  
plt.title("Comparativa Val_Accuracy en diferentes modelos")  
plt.legend(['vgg16','vgg19','ResNet152V2','DenseNet201','InceptionResNetV2','CNN_1'])
plt.savefig(directory_modelos + 'Comparativa_Val_Accuracy.png')  
#
plt.figure(4)  
plt.plot(vgg16_history['val_loss'],'r')
plt.plot(vgg19_history['val_loss'],'g')  
plt.plot(ResNet152V2_history['val_loss'],'b')  
plt.plot(DenseNet201_history['val_loss'],'y')  
plt.plot(InceptionResNetV2_history['val_loss'],'m')  
plt.plot(CNN_1_history['val_loss'],'c')  
plt.xticks(np.arange(0, EPOCHS, INTERVALO))
plt.rcParams['figure.figsize'] = (8, 6)  
plt.xlabel("Num of Epochs")  
plt.ylabel("Loss")  
plt.title("Comparativa Val_loss en diferentes modelos")  
plt.legend(['vgg16','vgg19','ResNet152V2','DenseNet201','InceptionResNetV2','CNN_1'])
plt.savefig(directory_modelos + 'Comparativa_Val_loss.png')  
#
plt.figure(5)  
plt.plot(vgg16_history['val_mse'],'r')
plt.plot(vgg19_history['val_mse'],'g')  
plt.plot(ResNet152V2_history['val_mse'],'b')  
plt.plot(DenseNet201_history['val_mse'],'y')  
plt.plot(InceptionResNetV2_history['val_mse'],'m')  
plt.plot(CNN_1_history['val_mse'],'c')  
plt.xticks(np.arange(0, EPOCHS, INTERVALO))
plt.rcParams['figure.figsize'] = (8, 6)  
plt.xlabel("Num of Epochs")  
plt.ylabel("MSE")  
plt.title("Comparativa Val_mse en diferentes modelos")  
plt.legend(['vgg16','vgg19','ResNet152V2','DenseNet201','InceptionResNetV2','CNN_1'])
plt.savefig(directory_modelos + 'Comparativa_Val_mse.png')  
#
plt.show()
#
print("[INFO]", 
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Fin del proceso",
    os.path.basename(__file__))