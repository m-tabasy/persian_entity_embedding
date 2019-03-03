from torch import optim as opt
from torch import nn

from entity2vec.model import Model
from entity2vec import train
from basic.commons import get_config, get_device, cuda_available

config = get_config()

# model params
ent_count = int(config['limit']['entity'])
vec_size = int(config['param']['vector_size'])
words_per_ent = int(config['param']['words_per_ent'])
neg_words = int(config['param']['neg_words'])

# training params
learning_rate = float(config['param']['lr'])
margin = float(config['param']['margin'])

# defining model

device = get_device()

network = Model(ent_count, vec_size, words_per_ent, neg_words).to(device)

loss_func = nn.MultiMarginLoss(p=1, margin=margin).to(device)

# SparseAdam, SGD work in cpu, gpu
# while AdaGrad only works on cpu (because of the embedding layer)
if cuda_available():
	print('-- cuda is available! so using adam optimizer.')
	optimizer = opt.Adam(network.parameters(), lr=learning_rate)
else:
	print('-- cuda is not available! so using adagrad optimizer.')
	optimizer = opt.AdaGrad(network.parameters(), lr=learning_rate)


if __name__ == '__main__':
	train.train_entity_vectors(network, loss_func, optimizer)
