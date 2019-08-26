#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), '.ipynb_checkpoints'))
	print(os.getcwd())
except:
	pass

#%%
#!/usr/bin/env python   esto es para ver cómo coger el directorio como parámetro inicializado. Del otro quitar comentarios


import glob
import random 
import uuid
import shutil
import os

target_dist = "color-80-20"


def processLine(_line):
	return (_line.split("\t")[0], _line.split("\t")[-1].strip())

_train = open("lmdb/"+target_dist+"/train.txt", "r")
TRAIN = []
for _line in _train.readlines():
	TRAIN.append( processLine(_line))	


_test = open("lmdb/"+target_dist+"/test.txt", "r")
TEST = []
for _line in _test.readlines():
	TEST.append( processLine(_line))


random.shuffle(TRAIN)
random.shuffle(TEST)

TRAIN_MAPPINGS = open("SVM/train_mapping.txt", "w")

percent_of_train = 0.2
for _entry in TRAIN[:int(percent_of_train*len(TRAIN))]:
	try:
		os.mkdir("SVM/train/"+_entry[-1]) #Try to create the label directory
	except:
		pass

	print "TRAIN :: Copying....", _entry
	oldName = _entry[0].replace("/home/mohanty/data/final_dataset/", "")
	newName = "SVM/train/"+_entry[-1]+"/"+str(uuid.uuid4()) + ".JPG"
	shutil.copy(oldName, newName)
	TRAIN_MAPPINGS.write(oldName+"\t"+newName + "\n")

TRAIN_MAPPINGS.close()	


TEST_MAPPINGS = open("SVM/test_mapping.txt", "w")

	
percent_of_test = 1
for _entry in TEST[:int(percent_of_test*len(TEST))]:
	try:
		os.mkdir("SVM/test/"+_entry[-1]) #Try to create the label directory
	except:
		pass

	print "TEST :: Copying....", _entry	
	oldName = _entry[0].replace("/home/mohanty/data/final_dataset/", "")
	newName = "SVM/test/"+_entry[-1]+"/"+str(uuid.uuid4()) + ".JPG"
	shutil.copy(oldName, newName)
	TEST_MAPPINGS.write(oldName+"\t"+newName + "\n")

TEST_MAPPINGS.close()


