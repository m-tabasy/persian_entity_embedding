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
	use_gpu = get_config().getboolean('param', 'gpu')
	
	if use_gpu and torch.cuda.is_available():
		return torch.device('cuda:0')
	else:
		return torch.device('cpu')


def show_progress(percent: float, title='', length=50, done=False, done_message='done!',
                  error=False, error_message='failed!'):
	if done:
		p = '=' * (length + 1)
		print(f'\r{title}[{p}] 100% {done_message}', flush=True)
	elif error:
		p = '=' * int(percent * length)
		r = '-' * (length - len(p))
		print(f'\r{title}[{p}X{r}] {(percent*100):.2f}% {error_message}', flush=True)
	else:
		p = '=' * int(percent * length)
		r = '-' * (length - len(p))
		print(f'\r{title}[{p}>{r}] {(percent*100):.2f}% ', end=' ', flush=True)


def get_hyperlinks_count():
	return 3006442
