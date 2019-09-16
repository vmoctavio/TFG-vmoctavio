# -----------------------------------------------------------------------------------------------
# Programa: TFGVOP_DataAugmentation_Process
# Autor: vmoctavio
# Fecha creación: 15/09/2019
# Descripción: Proceso para generar un Dataset sintético a partir de una estructura
#                   directory_root + plant_folder + plant_disease_folder
#              generando
#                   directory_dest + plant_folder + plant_disease_folder
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
# Devuelve una lista que contiene el nombre de las entradas en el directorio de la ruta del parámetro.
from os import listdir
#
# Manejo de arrays 
import numpy as np
#
# Usar comandos del sistema operativo
import os
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
def data_augmentation_image(image_dir):
#
# carga la imagen del parámtro, la convierte en un array NumPy y la remodela para incluir una dimensión más
    print("[INFO] Cargando imagen...",image_dir)
    image = load_img(image_dir)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

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

# Ejecución del generador de imágenes para el proceso "data augmentation" 
    imageGen = aug.flow(image, 
                        batch_size=1,           # número de imágenes por lote
                        save_to_dir=directory_dest+plant_folder+'/'+plant_disease_folder, # directorio destino
                        save_prefix="image",    # prefijo para usar para los nombres de archivo de las imágenes generadas
                        save_format="jpg")      # extensión de los archivos generados

# Bucle sobre el generador de imágenes
    for image in imageGen:
# Incrementar contador
        total += 1
	# Si hemos alcanzado el número total especificado, salimos del bucle
        if total == total_images:
            break   

