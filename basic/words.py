import os
import time
import torch
import pickle

from gensim.corpora import wikicorpus
from gensim.models import KeyedVectors
from hazm import word_tokenize, stopwords_list
from numpy.random import choice
from numpy import arange
from basic import commons

config = commons.get_config()

force_generate = config.getboolean('param', 'regenerate')
fa_wiki_path = config['path']['wiki']
word2vec_path = config['path']['word2vec']
words_path = config['path']['word']

# -------- globals --------

device = commons.get_device()

unk_word_token = 'UNK_W'  # not used :)
unk_word_id = -1

vocab_size = 0  # will be updated!

word2id, id2word = {}, {}

id2freq = torch.empty(vocab_size)  # will be created again! here vocab size is unknown!!
id2pow_freq = torch.empty(vocab_size)
id2doc_freq = torch.empty(vocab_size)

stop_words = []

word2vec = None


# -------- private by convention --------

def _load_word2vec(path, limit=1000):
	global word2vec
	
	print(f'-- loading {limit} word vectors from pretrained word2vec file at {path}...', end=' ', flush=True)
	
	word2vec = KeyedVectors.load_word2vec_format(word2vec_path, limit=limit)
	
	print('done!')


def _gen_freq(path, word2vec_based=True):
	global word2vec, word2id, id2word, id2freq, vocab_size, unk_word_id
	
	print(f'-- generating unigram based on {path} corpus...', end=' ', flush=True)
	
	_id2freq = {}
	
	word_count = unk_word_id + 1
	# word2id[unk_word_token] = unk_word_id
	# id2word[unk_word_id] = unk_word_token
	
	with open(path, 'r', encoding='utf8') as inf:
		
		for line in inf:
			clean_text = wikicorpus.filter_wiki(line)
			tokens = word_tokenize(clean_text)
			
			for token in tokens:
				# add new word, ignore if not found in word2vec vocab
				if token not in word2id.keys() and (not word2vec_based or token in word2vec.vocab.keys()):
					word2id[token] = word_count
					id2word[word_count] = token
					_id2freq[word_count] = 1
					word_count += 1
				
				elif token in word2id.keys():  # update frequency
					_id2freq[word2id[token]] += 1
	
	vocab_size = word_count
	
	# move freq from dict to torch array for performance
	id2freq = torch.empty(vocab_size, dtype=torch.float)
	for wid in _id2freq.keys():
		id2freq[wid] = _id2freq[wid]
	
	print('done!')


def _gen_powered_unigram(power=0.6):
	global id2freq, id2pow_freq
	
	print(f'-- generating powered unigram on {len(id2freq)} words...', end=' ', flush=True)
	
	id2freq = id2freq / id2freq.sum()
	
	id2pow_freq = id2freq ** power
	id2pow_freq = id2pow_freq / id2pow_freq.sum()
	
	print('done!')


def _init_words():
	global word2id, id2word, id2freq, id2pow_freq, id2doc_freq, vocab_size, unk_word_id, stop_words, words_path
	
	if vocab_size == 0:
		
		_load_word2vec(word2vec_path, limit=int(config['limit']['word']))
		
		if force_generate or not os.path.exists(words_path):
			
			if os.path.exists(words_path):
				print('-- forcing generate unigram!')
			else:
				print('-- generating unigram because no file was found!')
			
			_gen_freq(fa_wiki_path)
			_gen_powered_unigram(power=float(config['param']['unig_power']))
			_collect_stop_words(fa_wiki_path)
			
			with open(words_path, 'wb') as outf:
				pickle.dump(vocab_size, outf)
				pickle.dump(word2id, outf)
				pickle.dump(id2word, outf)
				pickle.dump(id2freq, outf)
				pickle.dump(id2pow_freq, outf)
				pickle.dump(id2doc_freq, outf)
				pickle.dump(stop_words, outf)
		else:
			
			print(f'-- loading unigram from {words_path}...', end=' ', flush=True)
			
			with open(words_path, 'rb') as inf:
				vocab_size = pickle.load(inf)
				word2id = pickle.load(inf)
				id2word = pickle.load(inf)
				id2freq = pickle.load(inf)
				id2pow_freq = pickle.load(inf)
				id2doc_freq = pickle.load(inf)
				stop_words = pickle.load(inf)
			
			print('done!')


# -------- public functions --------

def _collect_stop_words(path):
	global word2id, id2word, id2doc_freq, vocab_size, unk_word_id, stop_words
	
	print(f'-- detecting stop words based on {path} corpus...', end=' ', flush=True)
	
	id2doc_freq = torch.zeros(vocab_size, dtype=torch.int)
	
	current_doc = ''
	doc_count = 0
	
	with open(path, 'r', encoding='utf8') as inf:
		
		for line in inf:
			
			if line.startswith('<doc'):
				words = set(word_tokenize(wikicorpus.filter_wiki(current_doc)))
				for word in words:
					if word in word2id.keys():
						word_id = word2id[word]
						id2doc_freq[word_id] += 1
				current_doc = ''
				doc_count += 1
			
			else:
				current_doc += line
	
	for word_id in range(vocab_size):
		if id2doc_freq[word_id] > doc_count / 3:
			stop_words.append(id2word[word_id])
	
	print('done!')


def get_stop_words() -> set:
	return set(stopwords_list() + stop_words + ['.', '،', ':', '(', ')', '«', '»', '–', '"'])


# -------- test --------

def get_unig_sample(n: int, powered=True):
	# found no weighted sampling in torch, so using numpy
	
	if powered:
		global id2pow_freq
		return torch.from_numpy(choice(arange(vocab_size), size=n, p=id2pow_freq.numpy())).long()
	else:
		global id2freq
		return torch.from_numpy(choice(arange(vocab_size), size=n, p=id2freq.numpy())).long()


if __name__ == '__main__':
	
	# check init time
	_start = time.time()
	_init_words()
	_end = time.time()
	print(f'-- calculated time is {_end - _start} sec')
	
	# getting sample words with powered unigram prob
	sampled_ids = get_unig_sample(20)
	sampled_words = list(map(lambda id: id2word[id], sampled_ids.data.tolist()))
	print(sampled_words)
	
	# checking collected stop words
	print(stop_words)
	pass

# -------- load when imported --------

else:
	_init_words()
