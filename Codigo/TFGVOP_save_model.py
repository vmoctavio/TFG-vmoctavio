# -----------------------------------------------------------------------------------------------
# Programa: TFGVOP_save_model
# Autor: vmoctavio
# Fecha creación: 12/10/2019
# Descripción: Función para guardar las estadísticas de un modelo
#              Parámetors de entrda:
#                   directory_log: Ruta donde se guardarán los datos del modelo de entrada
#                   file_name: Nombre del fichero a guardar
#                   model: Modelo a guardar
#                   v_Registro: Datos a guardar
# -----------------------------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------------------------
# Imports necesarios
# -----------------------------------------------------------------------------------------------
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
# Librería para gestionar archivos csv
import csv
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
# Proceso principal: guarda las estadísticas del modelo "model" 
# en un archivo con nombre "file_name"
# -----------------------------------------------------------------------------------------------
#
def save_model(directory_log,file_name,model,v_Registro):
#
    try:
        #creamos el directorio destino si no existe 
        try:
            os.stat(directory_log + 'modelos')
        except:
            os.mkdir(directory_log + 'modelos')
#
        with open(directory_log + 'modelos/' + file_name + '.stats', 'wb') as file_modelo:  
            pickle.dump(model.history, file_modelo)
#
        print("[INFO]", 
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Modelo correctamente guardado... ",
            file_name)
#
    except Exception as e: # Controla cualquier error que se produzca en la función
        print("[ERROR]",
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            sys.exc_info()[0],
            sys.exc_info()[1],
            sys.exc_info()[2])
#
    try:
        fieldnames = ['Network',
                    'Hora Inicio Proceso',
                    'Hora Fin Proceso',
                    'Duración Proceso',
                    'Hora Inicio Precisión',
                    'Hora Fin Precisión',
                    'Duración Precisión',
                    'EPOCHS',
                    'INIT_LR',
                    'BATCH_SIZE',
                    'width',
                    'height',
                    'depth',
                    'TEST_SIZE',
                    'VALID_SIZE',
                    'RANDOM_STATE',
                    'VERBOSE',
                    'CLIPVALUE',
                    'ACCURACY',
                    'LOSS',
                    'TOTAL_IMAGES',
                    'TOTAL_LABEL',
                    'loss_function',
                    'Etiquetas correctas',
                    'Etiquetas incorrectas',
                      ]
#       creamos el fichero si no existe 
        if ( not os.path.isfile(directory_log + 'modelos/' + 'results_data.csv')):
            with open(directory_log + 'modelos/' + 'results_data.csv', 'w') as file_data:  
                writer = csv.DictWriter(file_data, fieldnames=fieldnames)
                writer.writeheader()
                file_data.close()                
#       guardamos los resultados
        with open(directory_log + 'modelos/' + 'results_data.csv', 'a') as file_data:
            writer = csv.DictWriter(file_data, fieldnames=fieldnames)
            writer.writerows([v_Registro])               
            file_data.close() 
#
        print("[INFO]", 
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Datos correctamente guardados... ",
            'results_data.csv')
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
    # Fin de la ejecución
    # -----------------------------------------------------------------------------------------------
    #
    return
#
# -----------------------------------------------------------------------------------------------
# Fin del proceso
# -----------------------------------------------------------------------------------------------
#
