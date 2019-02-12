import configparser
import torch
import os

from torch import optim as opt
from torch import nn

from entity2vec.model import Model
from entity2vec import train

config = configparser.ConfigParser()
file_path = os.path.dirname(__file__)
config_path = os.path.join(file_path, 'config.ini')
config.read(config_path)

# model params
ent_count = int(config['limit']['entity'])
vec_size = int(config['param']['vector_size'])
words_per_ent = int(config['param']['words_per_ent'])
neg_words = int(config['param']['neg_words'])

# training params
learning_rate = float(config['param']['lr'])
margin = float(config['param']['margin'])

# defining model

device = torch.device('cuda:0')

network = Model(ent_count, vec_size, words_per_ent, neg_words)    # .cuda() do the same for inputs & targets...

loss_func = nn.MultiMarginLoss(p=1, margin=margin)

# SparseAdam, SGD work in cpu, gpu
# while AdaGrad only works on cpu (because of the embedding layer)
optimizer = opt.Adagrad(network.parameters(), lr=learning_rate)

if __name__ == '__main__':
	train.train_entity_vectors(network, loss_func, optimizer)
