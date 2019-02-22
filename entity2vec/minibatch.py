import time
import torch
from multiprocessing import Pool
from threading import Thread
from hazm import word_tokenize

from basic.words import get_unig_sample, word2id, unk_word_id
from basic.entities import wikiid2id

from basic import commons

config = commons.get_config()
device = commons.get_device()

canonical_words_path = config['path']['canonical']
hyp_ctx_words_path = config['path']['hyp_ctx']

batch_size = int(config['param']['batch_size'])
passes_wiki_words = int(config['param']['passes_wiki_words'])
words_per_ent = int(config['param']['words_per_ent'])
neg_words = int(config['param']['neg_words'])

current_data = canonical_words_path
current_file = open(current_data, 'r', encoding='utf8')
current_pass = 1

_times = [0, 0, 0, 0, 0, 0, 0]
_t = 0

def read_one_line():
	global times
	global canonical_words_path, hyp_ctx_words_path, passes_wiki_words
	global current_data, current_file, current_pass
	
	_t = time.time()

	line = current_file.readline()

	if not line:  # end of current file

		current_file.close()

		print('done!')
		
		if current_data == hyp_ctx_words_path:
			return # means all passes are done. go to next epoch
		
		elif current_pass == passes_wiki_words:
			current_data = hyp_ctx_words_path
			current_pass = 0


		current_pass += 1

		print(f'-- starting {current_pass}-th pass on {current_data}.', end=' ', flush=True)

		current_file = open(current_data, 'r', encoding='utf8')
		line = current_file.readline()
		
	_times[0] += time.time() - _t

	return line


def process_one_line(line, minibatch, minitargets, batch_idx):
	global _times
	_t = time.time()
	
	line = line[:-1]    # removing \n at end of line
	
	if line.find('CNC_WORDS') > 0:  # wiki canonical page words
		
		header, words = line.split('\tCNC_WORDS')
		ent_wikiid, ent_name = header.split('\t')
		pos_word_ids = words.split('\t')
	
	else:   # wiki hyperlink context word
		
		header, words = line.split('\tLEFT_CTX\t')
		ent_wikiid, ent_name, mention = header.split('\t')
		left_context = words.split('RIGHT_CTX')[0].split('\t')
		right_context = words.split('RIGHT_CTX')[1].split('\t')
		pos_word_ids = left_context + right_context
	
	_times[1] += time.time() - _t
	_t = time.time()
	
	ent_wikiid = int(ent_wikiid)
	minibatch['entity_id'][batch_idx] = wikiid2id[ent_wikiid]
	# minibatch['entity_wikiid'][batch_idx] = ent_wikiid
	
	pos_word_ids = list(filter(lambda wid: len(wid) > 0, pos_word_ids))
	pos_word_ids = list(map(lambda wid: int(wid), pos_word_ids))
	pos_word_ids = torch.LongTensor(pos_word_ids).to(device)
	
	_times[2] += time.time() - _t
	_t = time.time()
	
	# pos words are in vocab and not stop words from data gen step
	# use entity name if no positive context words available
	if len(pos_word_ids) == 0:
		entity_name_words = word_tokenize(ent_name)
		entity_name_words = list(filter(lambda w: w in word2id.keys(), entity_name_words))
		pos_word_ids = list(map(lambda w: word2id[w], entity_name_words))
		pos_word_ids = torch.LongTensor(pos_word_ids).to(device)
	
	# get random sample if still empty
	if len(pos_word_ids) == 0:
		pos_word_ids = get_unig_sample(10)
	
	_times[3] += time.time() - _t
	_t = time.time()
	
	# fill with negative words sampled from powered unigram
	neg_sample = get_unig_sample(words_per_ent * neg_words).view(words_per_ent, neg_words)
	minibatch['word_ids'][batch_idx] = neg_sample
	
	# sample some positive words, place them somewhere between negative words
	# and store that index in targets
	pos_sample_indexes = torch.randint(high=len(pos_word_ids), size=(words_per_ent,), device=device)
	targets = torch.randint(high=neg_words, size=(words_per_ent,), device=device)
	minitargets[batch_idx] = targets
	
	_times[4] += time.time() - _t
	_t = time.time()
	
	flat_pos_indexes = (torch.arange(0, words_per_ent, device=device) * neg_words) + targets
	minibatch['word_ids'][batch_idx].put_(flat_pos_indexes.long(), pos_word_ids[pos_sample_indexes])

	_times[5] += time.time() - _t
	_t = time.time()
	

def gen_minibatch():
	global _times, thread_count, pool
	global batch_size, passes_wiki_words, words_per_ent, neg_words
	global canonical_words_path, hyp_ctx_words_path, passes_wiki_words
	global current_data, current_file, current_pass

	print(f'-- starting first pass.', end=' ', flush=True)

	current_data = canonical_words_path
	current_file = open(current_data, 'r', encoding='utf8')
	current_pass = 1

	data_available = True

	while data_available:
		
		_t = time.time()

		minibatch = {'word_ids': torch.ones(batch_size, words_per_ent, neg_words, device=device, dtype=torch.long ).mul(unk_word_id),
		             'entity_id': torch.ones(batch_size, device=device, dtype=torch.long)}

		targets = torch.zeros(batch_size, words_per_ent, device=device, dtype=torch.long)
		
		_times[6] += time.time() - _t

		for batch_index in range(batch_size): # // thread_count):
			
			line = read_one_line()
			
			if line:
				process_one_line(line, minibatch, targets, batch_index)
			else:
				data_available = False
				break

		if data_available:

			minibatch['word_ids'] = minibatch['word_ids'].view(batch_size, words_per_ent * neg_words)
			targets = targets.view(batch_size * words_per_ent)

			yield minibatch, targets


if __name__ == '__main__':
	
	_start = time.time()
	
	batch_count = 0
	for inputs, targets in gen_minibatch():
		batch_count += 1
		print(_times)
	
	_end = time.time()
	print(f'-- calculated time is {_end - _start} sec')



