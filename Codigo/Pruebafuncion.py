#
import TFGVOP_load_dataset
#
#from TFGVOP_load_dataset.py import load_dataset_process
#
# Directorio de trabajo donde está el dataset
directory_root = '/Users/vmoctavio/Downloads/keras-tutorial-dataaugmentation/Dataset_vid_prueba_dest/'
# Directorio de trabajo donde está el dataset
directory_root = '/Users/vmoctavio/Downloads/keras-tutorial-dataaugmentation/Dataset_vid_prueba_dest/'
image_list=[]
image_labels=[]
labels=[]
(image_labels,image_list,labels) = TFGVOP_load_dataset.load_dataset_process(directory_root)

print("labels: ",labels)
print("image_labels después de llamar: ", image_labels)
#print("image_list: ",image_list)
