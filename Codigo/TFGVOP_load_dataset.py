# -----------------------------------------------------------------------------------------------
# Programa: TFGVOP_load_dataset
# Autor: vmoctavio
# Fecha creación: 03/10/2019
# Descripción: Función para cargar un Dataset a partir de un determinadao directorio.
#              Parámetors de entrda:
#                   directory_root: ruta a partir de la cual está el dataset organizado en carpetas
#                                   (una por etiqueta)
#              Parámetros de salida:
#                   label_list: array de array's de etiquetas
#                   image_list: array de array's resultado de convertir las imágenes del directorio
#                   labels: array con las distintas etiquetas del dataset
# -----------------------------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------------------------
# Imports necesarios
# -----------------------------------------------------------------------------------------------
#
# Convertir una imagen en una matriz NumPy 
from keras.preprocessing.image import img_to_array
#
# Usar comandos del sistema operativo ver si esto se borra
import os
#
# Devuelve una lista que contiene el nombre de las entradas en el directorio de la ruta del parámetro.
from os import listdir
#
# Módulo para control de excepciones
import sys
#
# Manejo de arrays 
import numpy as np
#
# Para manipular fechas y horas
import datetime
#
# Librería para el procesamiento de imágenes
import cv2
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
    "Leyendo archivo de configuración en ",
    os.path.basename(__file__))
#
config = configparser.ConfigParser()
config.read('TFGVOP_Config.ini')
#
# Tamaño de la imagen: ancho
width = int(config.get('config','width'))
# Tamaño de la imagen: alto
height = int(config.get('config','height'))
# Número de registros a cargar por directorio
CONTREG = int(config.get('config','CONTREG'))
#
# -----------------------------------------------------------------------------------------------
# Inicialización de variables
# -----------------------------------------------------------------------------------------------
#
# Tamaño por defecto de las imágenes
default_image_size = tuple((height, width))
#
# -----------------------------------------------------------------------------------------------
# Función para redimensionar una imagen y convertirla en array
# -----------------------------------------------------------------------------------------------
#
def convert_image_to_array(image_dir):
    try:
        # Carga la imagen del fichero del parámetro
        image = cv2.imread(image_dir)
        if image is not None:   # Detecta si el fichero existe
            # Redimensiona la imagen al tamaño especificado en la variable default_image_size
            image = cv2.resize(image, default_image_size)
            return img_to_array(image) # Devuelve como array la imagen del parámetro
        else :
            return np.array([]) # Devuelve un array vacío
    except Exception as e: # Controla cualquier error que se produzca en la función
        print("[ERROR]", 
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            sys.exc_info()[0],
            sys.exc_info()[1],
            sys.exc_info()[2])
        return None
#
# -----------------------------------------------------------------------------------------------
# Proceso principal: lee las carpetas del directorio y genera los correspondientes arrays para
#                    imágenes y etiquetas
# -----------------------------------------------------------------------------------------------
#
def load_dataset_process(directory_root):
#   
    cont_labels = 0         # Contador de etiquetas diferentes
    image_label = []        # Array con el nombre de las etiquetas
    labels = []             # Array con las distintas etiquetas del dataset
    label_list = []         # Array de Array's de etiquetas
    image_list = []         # Array de Array's de imágenes
#
    try:
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
            print(f"[INFO]", 
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Procesando el directorio... ",
                plant_disease_folder)
#
            total_images_origen = 0 # contador imágenes por cada directorio
#
            labels.append(plant_disease_folder)   # Lista con las diferentes etiquetas
            cont_labels = cont_labels + 1
#        
            # Genera una lista con todos los archivos de un directorio concreto
            plant_disease_image_list = listdir(f"{directory_root}/{plant_disease_folder}/")
# 
            for single_plant_disease_image in plant_disease_image_list:  # Nos recorremos todos los archivos de un directorio concreto
                # elimina .DS_Store de la lista en caso de existir
                if single_plant_disease_image == ".DS_Store" :
                    plant_disease_image_list.remove(single_plant_disease_image)
#
            for image in plant_disease_image_list[:CONTREG]: # Nos recorremos todos los archivos de un directorio concreto
                image_directory = f"{directory_root}/{plant_disease_folder}/{image}"
                # Si el archivo tiene extensión jpg o JPG lo tratamos 
                if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:
                    image_list.append(convert_image_to_array(image_directory))  # Convierte imagen a array
                    image_label.append(cont_labels-1)
                    total_images_origen = total_images_origen + 1 #  contador imágenes por cada directorio
#
            print("[INFO]", 
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Total imágenes del directorio ... ",
                plant_disease_folder,total_images_origen)
#
        print("[INFO]",
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Proceso de carga de imágenes completado ")
#
        print("[INFO]",
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Total Etiquetas... ",
            cont_labels)
#
    except Exception as e: # Controla cualquier error que se produzca en la función
        print("[ERROR]",
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            sys.exc_info()[0],
            sys.exc_info()[1],
            sys.exc_info()[2])
    #
    print("[INFO]", 
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Fin del proceso",
        os.path.basename(__file__))
    #
    # -----------------------------------------------------------------------------------------------
    # Fin de la ejecución y devolución del resultado en los parámetros de salida
    # -----------------------------------------------------------------------------------------------
    #
    return(image_label,image_list,labels)
#
# -----------------------------------------------------------------------------------------------
# Fin del proceso
# -----------------------------------------------------------------------------------------------
#
