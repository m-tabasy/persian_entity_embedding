from torch import nn
import numpy as np

from basic.words import *


def get_word_vectors():
	global word2vec, id2word, vocab_size
	
	word_vectors = word2vec.vectors
	vocab = word2vec.vocab
	new_indexes = np.empty(vocab_size, dtype=np.int_)
	
	for word_id in range(vocab_size):
		word = id2word[word_id]
		new_indexes[word_id] = vocab[word].index
	
	return word_vectors[new_indexes]


class Model(nn.Module):
	
	def __init__(self, ent_count, vec_size, words_per_ent, neg_words):
		super(Model, self).__init__()
		
		self.embedding = nn.Embedding(ent_count, vec_size, max_norm=1, norm_type=2)
		
		pretrained_word_vectors = torch.from_numpy(get_word_vectors())
		self.word2vec = nn.Embedding.from_pretrained(embeddings=pretrained_word_vectors, freeze=True)
		
		self.ent_count = ent_count
		self.vec_size = vec_size
		self.words_per_ent = words_per_ent
		self.neg_words = neg_words
	
	def forward(self, x):
		word_ids = x[0]
		word_vectors = self.word2vec(word_ids)
		word_vectors = nn.functional.normalize(word_vectors)
		word_vectors = word_vectors.view(-1, self.words_per_ent * self.neg_words, self.vec_size)
		
		entity_ids = x[1]
		entity_vectors = self.embedding(entity_ids)
		# 		entity_vectors = nn.functional.normalize(entity_vectors)
		entity_vectors = entity_vectors.view(-1, 1, self.vec_size)
		
		cosines = torch.matmul(word_vectors, entity_vectors.transpose(1, 2))
		output = cosines.view(-1, self.neg_words)
		
		return output
	
	def get_entity_vector(self, ent_id):
		return self.embedding(ent_id)
