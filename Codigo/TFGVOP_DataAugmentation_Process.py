# -----------------------------------------------------------------------------------------------
# Programa: TFGVOP_DataAugmentation_Process
# Autor: vmoctavio
# Fecha creación: 15/09/2019
# Descripción: Proceso para generar un Dataset sintético a partir de un determinado directorio,
#              en el que como subdirectorios están las diferentes carpetas, una por etiqueta, con
#              los archivos de imágenes.
#              Las variables a inicializar en el proceso son:
#                   directory_root: ruta a partir de la cual está el dataset organizado en carpetas
#                   directory_dest: ruta en la que se generará el nuevo dataset
#                   total_images: número de imágenes a aumentar
# -----------------------------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------------------------
# Imports necesarios
# -----------------------------------------------------------------------------------------------
#
# Aumentar datos en tipo real 
from keras.preprocessing.image import ImageDataGenerator
#
# Convertir una imagen en una matriz NumPy 
from keras.preprocessing.image import img_to_array
#
# Cargar una imagen  
from keras.preprocessing.image import load_img
#
# Usar comandos del sistema operativo
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
# -----------------------------------------------------------------------------------------------
# Inicio del proceso
# -----------------------------------------------------------------------------------------------
print("[INFO]",
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Inicio del proceso TFGVOP_DataAugmentation_Process ")
#
# -----------------------------------------------------------------------------------------------
# Inicialización de variables
# -----------------------------------------------------------------------------------------------
#
# Directorio de trabajo donde está el dataset
directory_root = '/Users/vmoctavio/Downloads/keras-tutorial-dataaugmentation/Dataset_vid_prueba/'
# Directorio de trabajo donde se dejará el dataset tras proceso data augmentation
directory_dest = '/Users/vmoctavio/Downloads/keras-tutorial-dataaugmentation/Dataset_vid_prueba_dest/'
# Número total de imágenes a generar a partir de una determinada
total_images = 20
#
# -----------------------------------------------------------------------------------------------
# Función que genera "total_images" a partir de una dada
# -----------------------------------------------------------------------------------------------
#
def data_augmentation_image(image_name,image_dir):
#
# carga la imagen del parámtro, la convierte en un array NumPy y la remodela para incluir una dimensión más
    image = load_img(image_dir)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
#
# Inicializa el generador de imágenes para el proceso "data augmentation" 
    aug = ImageDataGenerator(rotation_range=180,                   # grados [-XX, +XX] para rotación aleatoria
                             zoom_range=0.2,                       # cantidad de zoom [1-XX, 1+XX]
                             width_shift_range=0.2,                # desplazamiento en anchura [-XX, +XX]
                             height_shift_range=0.2,               # desplazamiento en altura [-XX, +XX]
                             shear_range=0.15,                     # intensidad de corte para deformar la forma de los objetos
                             horizontal_flip=True,                 # voltear horizontalmente
                             vertical_flip=True,                   # voltear verticalmente
                             fill_mode="nearest")                  # los píxeles de fuera de los límites de la entrada se rellenan con los píxeles más cercanos
# Inicializa el contador de imágenes a generar
    total = 0
#
# Ejecución del generador de imágenes para el proceso "data augmentation" 
    imageGen = aug.flow(image, 
                        batch_size=1,           # número de imágenes por lote
                        save_to_dir=directory_dest+plant_disease_folder, # directorio destino
                        save_prefix=image_name, # prefijo para usar para los nombres de archivo de las imágenes generadas
                        save_format="jpg")      # extensión de los archivos generados
#
# Bucle sobre el generador de imágenes
    for image in imageGen:
        total += 1                  # Incrementar contador
        if total == total_images:   # Si hemos alcanzado el número total especificado, salimos del bucle
            break   
#
# -----------------------------------------------------------------------------------------------
# Proceso principal: bucle que lee las carpetas del directorio del parámetros
#                    y los archivos contenidos en ellas
# -----------------------------------------------------------------------------------------------
#
try:
    root_dir = listdir(directory_root)      # Genera una lista con los directorios existentes
#
    for directory in root_dir :
        # elimina .DS_Store de la lista en caso de existir
        if directory == ".DS_Store" :
            root_dir.remove(directory)
#
#   creamos el directorio destino si no existe 
    try:
        os.stat(directory_dest)
    except:
        os.mkdir(directory_dest)
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
        #creamos el directorio destino si no existe 
        try:
            os.stat(directory_dest+plant_disease_folder)
        except:
            os.mkdir(directory_dest+plant_disease_folder)
# 
        for single_plant_disease_image in plant_disease_image_list:  # Nos recorremos todos los archivos de un directorio concreto
            # elimina .DS_Store de la lista en caso de existir
            if single_plant_disease_image == ".DS_Store" :
                plant_disease_image_list.remove(single_plant_disease_image)

        for image in plant_disease_image_list: # Nos recorremos todos los archivos de un directorio concreto
            image_directory = f"{directory_root}/{plant_disease_folder}/{image}"
            # Si el archivo a tratar tiene extensión jpg o JPG se ejecuta "dataaugmentation" 
            if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:
#                print("[INFO]", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Cargando imagen ... ",image)
                data_augmentation_image(os.path.splitext(image)[0],image_directory)
                total_images_origen = total_images_origen + 1 #  contador imágenes por cada directorio
#
        print("[INFO]",
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Total imágenes del directorio ... ",
            plant_disease_folder,total_images_origen)
#
    print("[INFO]",
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Fin del proceso TFGVOP_DataAugmentation_Process ")
#
except Exception as e: # Controla cualquier error que se produzca en el proceso
    print("[ERROR]",
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        sys.exc_info()[0],
        sys.exc_info()[1],
        sys.exc_info()[2])
    print("[INFO]",
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Fin del proceso TFGVOP_DataAugmentation_Process ")
#
# -----------------------------------------------------------------------------------------------
# Fin del proceso
# -----------------------------------------------------------------------------------------------
#


