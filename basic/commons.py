import configparser
import os

import torch


def get_config():
	
	config = configparser.ConfigParser()
	file_path = os.path.dirname(__file__)
	config_path = os.path.join(file_path, '../config.ini')
	config.read(config_path)
	
	return config


def get_device():
	
	if torch.cuda.is_available() and False:  # TODO: fix this
		return torch.device('cuda:0')
	else:
		return torch.device('cpu')