#!/usr/local/bin/python

import configparser
config = configparser.ConfigParser()
config.read('TFGVOP_Config.ini')
print('directory_root',config.get('config','directory_root'))
print('EPOCHS',config.get('config','EPOCHS'))
print('INIT_LR',config.get('config','INIT_LR'))
print('BATCH_SIZE',config.get('config','BATCH_SIZE'))
print('width',config.get('config','width'))
print('height',config.get('config','height'))
print('depth',config.get('config','depth'))
print('TEST_SIZE',config.get('config','TEST_SIZE'))
print('VALID_SIZE',config.get('config','VALID_SIZE'))
print('RANDOM_STATE',config.get('config','RANDOM_STATE'))
print('directory_log',config.get('config','directory_log'))
print('directory_modelos',config.get('config','directory_modelos'))