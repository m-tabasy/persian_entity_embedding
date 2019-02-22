import configparser
import os

import torch


def get_config():
	
	config = configparser.ConfigParser()
	file_path = os.path.dirname(__file__)
	config_path = os.path.join(file_path, '../config.ini')
	config.read(config_path)
	
	return config


def cuda_available():
	return torch.cuda.is_available()


def get_device():
	
	use_gpu = get_config()['param']['gpu']
	
	if use_gpu and torch.cuda.is_available():
		return torch.device('cuda:0')
	else:
		return torch.device('cpu')
