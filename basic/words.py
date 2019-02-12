import configparser
import torch
import os

from gensim.corpora import wikicorpus
from gensim.models import KeyedVectors
from hazm import word_tokenize, stopwords_list
from numpy.random import choice

config = configparser.ConfigParser()
file_path = os.path.dirname(__file__)
config_path = os.path.join(file_path, '../config.ini')
config.read(config_path)

fa_wiki_path = config['path']['wiki']
word2vec_path = config['path']['word2vec']

# -------- globals --------

if torch.cuda.is_available() and False:
	device = torch.device('cuda:0')
else:
	device = torch.device('cpu')

unk_word_token = 'UNK_W'
unk_word_id = -1
vocab_size = 0  # will be updated!

word2id = {}
id2word = {}
id2freq = {}
id2pow_freq = {}
id2doc_freq = {}
word2vec = None

stop_words = []


def load_word2vec(path, limit=1000):
	global word2vec

	print(f'-- loading {limit} word vectors from pretrained word2vec file at {path}...', end=' ', flush=True)

	word2vec = KeyedVectors.load_word2vec_format(word2vec_path, limit=limit)

	print('done!')


def gen_freq(path, word2vec_based=True):
	global word2vec, word2id, id2word, id2freq, vocab_size, unk_word_id

	print(f'-- generating unigram based on {path} corpus...', end=' ', flush=True)

	word_count = unk_word_id + 1
	# word2id[unk_word_token] = unk_word_id
	# id2word[unk_word_id] = unk_word_token

	with open(path, 'r',encoding='utf8') as inf:

		for line in inf:
			clean_text = wikicorpus.filter_wiki(line)
			tokens = word_tokenize(clean_text)

			for token in tokens:
				# add new word, ignore if not found in word2vec vocab
				if token not in word2id.keys() and (not word2vec_based or token in word2vec.vocab.keys()):
					word2id[token] = word_count
					id2word[word_count] = token
					id2freq[word_count] = 1
					word_count += 1

				elif token in word2id.keys():   # update frequency
					id2freq[word2id[token]] += 1

	vocab_size = word_count
	print('done!')


def gen_powered_unigram(power=0.6):
	global id2freq, id2pow_freq

	print(f'-- generating powered unigram on {len(id2freq.keys())} words...', end=' ', flush=True)

	freq_sum = sum(id2freq.values())

	for key in id2freq.keys():
		id2freq[key] /= freq_sum
		id2pow_freq[key] = float(id2freq[key]) ** power

	pow_freq_sum = sum(id2pow_freq.values())

	for key in id2pow_freq.keys():
		id2pow_freq[key] /= pow_freq_sum

	print('done!')


def get_unig_sample(n: int, powered=True):

	# found no weighted sampling in torch, so using numpy

	if powered:
		global id2pow_freq
		return torch.from_numpy(choice(list(id2pow_freq.keys()), size=n, p=list(id2pow_freq.values()))).to(device)
	else:
		global id2freq
		return torch.from_numpy(choice(list(id2freq.keys()), size=n, p=list(id2freq.values()))).to(device)


def collect_stop_words(path):
	global word2id, id2word, id2doc_freq, vocab_size, unk_word_id, stop_words

	print(f'-- detecting stop words based on {path} corpus...', end=' ', flush=True)

	for i in range(vocab_size):
		id2doc_freq[i] = 0

	current_doc = ''
	doc_count = 0

	with open(path, 'r',encoding='utf8') as inf:
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

	for word_id, freq in id2doc_freq.items():
		if freq > doc_count / 3:
			stop_words.append(id2word[word_id])

	print('done!')


def get_stop_words():
	return set(stopwords_list() + stop_words + ['.', '،', ':', '(', ')', '«', '»', '–', '"'])


# -------- generate unigram --------

if len(id2pow_freq) == 0:
	load_word2vec(word2vec_path, limit=int(config['limit']['word']))
	gen_freq(fa_wiki_path)
	gen_powered_unigram(power=float(config['param']['unig_power']))

# -------- test --------

if __name__ == '__main__':

	# getting sample words with powered unigram prob
	sampled_ids = get_unig_sample(20)
	sampled_words = list(map(lambda id: id2word[id], sampled_ids))
	print(sampled_words)

	collect_stop_words(fa_wiki_path)
	print(stop_words)
