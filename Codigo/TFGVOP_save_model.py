# -----------------------------------------------------------------------------------------------
# Programa: TFGVOP_save_model
# Autor: vmoctavio
# Fecha creación: 12/10/2019
# Descripción: Función para guardar las estadísticas de un modelo
#              Parámetors de entrda:
#                   directory_log: Ruta donde se guardarán los datos del modelo de entrada
#                   model_name: Nombre del modelo a guardar
#                   model: Modelo a guardar
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
# en un archivo con nombre "model_name"
# -----------------------------------------------------------------------------------------------
#
def save_model(directory_log,model_name,model):
#
    try:
        #creamos el directorio destino si no existe 
        try:
            os.stat(directory_log + 'modelos')
        except:
            os.mkdir(directory_log + 'modelos')
#
        with open(directory_log + 'modelos/' + model_name + '.stats', 'wb') as file_modelo:  
            pickle.dump(model.history, file_modelo)
#
        print("[INFO]", 
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Modelo correcamente guardado... ",
            model_name)
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
