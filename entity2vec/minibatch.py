import re
import time
import torch
from hazm import word_tokenize

from basic.commons import show_progress, get_hyperlinks_count
from basic.words import get_unig_sample, word2id, unk_word_id, vocab_size
from basic.entities import wikiid2id, ent_size

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
line_index = 0

_times = [0, 0, 0, 0, 0, 0, 0, 0]
_t = 0


def read_one_line():
	global times
	global canonical_words_path, hyp_ctx_words_path, passes_wiki_words
	global current_data, current_file, current_pass, line_index
	
	_t = time.time()
	
	line = current_file.readline()
	line_index += 1
	
	if not line:  # end of current file
		
		current_file.close()
		line_index = 0
		
		show_progress(1.0, title='   ', done=True)
		
		if current_data == hyp_ctx_words_path:
			return  # means all passes are done. go to next epoch
		
		elif current_pass == passes_wiki_words:
			current_data = hyp_ctx_words_path
			current_pass = 0
		
		current_pass += 1
		
		if current_data == canonical_words_path:
			print(f'-- pass #{current_pass} on canonical pages:')
		else:
			print(f'-- passing through hyperlink contexts:')
		
		current_file = open(current_data, 'r', encoding='utf8')
		line = current_file.readline()
	
	_times[0] += time.time() - _t
	
	return line


def process_one_line(line, word_ids, entity_ids, minitargets, batch_idx):
	global _times
	_t = time.time()
	
	line = line[:-1]  # removing \n at end of line
	
	if line.find('CNC_WORDS') > 0:  # wiki canonical page words
		
		header, words = line.split('\tCNC_WORDS')
		ent_wikiid, ent_name = header.split('\t')
		pos_word_ids = words.split('\t')
	
	else:  # wiki hyperlink context words
		
		header, words = line.split('\tLEFT_CTX\t')
		ent_wikiid, ent_name, mention = header.split('\t')
		left_context = words.split('RIGHT_CTX')[0].split('\t')
		right_context = words.split('RIGHT_CTX')[1].split('\t')
		pos_word_ids = left_context + right_context
	
	_times[1] += time.time() - _t
	_t = time.time()
	
	ent_wikiid = int(ent_wikiid)
	entity_ids[batch_idx] = wikiid2id[ent_wikiid]
	
	pos_word_ids = filter(lambda wid: len(wid) > 0, pos_word_ids)
	pos_word_ids = list(map(lambda wid: int(wid), pos_word_ids))
	pos_word_ids = torch.LongTensor(pos_word_ids)
	
	_times[2] += time.time() - _t
	_t = time.time()
	
	# pos words are in vocab and not stop words from data gen step
	# use entity name if no positive context words available
	if len(pos_word_ids) == 0:
		entity_name_words = word_tokenize(ent_name)
		entity_name_words = list(filter(lambda w: w in word2id, entity_name_words))
		pos_word_ids = list(map(lambda w: word2id[w], entity_name_words))
		pos_word_ids = torch.LongTensor(pos_word_ids)
	
	# get random sample if still empty
	if len(pos_word_ids) == 0:
		pos_word_ids = get_unig_sample(10)
	
	_times[3] += time.time() - _t
	_t = time.time()
	
	# sample some positive words, place them somewhere between negative words
	# and store that index in targets
	pos_sample_indexes = torch.randint(high=len(pos_word_ids), size=(words_per_ent,))
	targets = torch.randint(high=neg_words, size=(words_per_ent,))
	minitargets[batch_idx] = targets
	
	_times[4] += time.time() - _t
	_t = time.time()
	
	flat_pos_indexes = (torch.arange(0, words_per_ent) * neg_words) + targets
	word_ids[batch_idx].put_(flat_pos_indexes.long(), pos_word_ids[pos_sample_indexes])
	
	_times[5] += time.time() - _t
	_t = time.time()


def gen_minibatch():
	global _times
	global batch_size, passes_wiki_words, words_per_ent, neg_words
	global canonical_words_path, hyp_ctx_words_path, passes_wiki_words
	global current_data, current_file, current_pass, line_index
	
	print(f'-- pass #1 on canonical pages:')
	
	current_data = canonical_words_path
	current_file = open(current_data, 'r', encoding='utf8')
	current_pass = 1
	
	data_available = True
	
	while data_available:
		_t = time.time()
		
		# fill with negative words sampled from powered unigram
		word_ids = get_unig_sample(batch_size * words_per_ent * neg_words).view(batch_size, words_per_ent, neg_words)
		entity_ids = torch.ones(batch_size, dtype=torch.long)
		targets = torch.zeros(batch_size, words_per_ent, dtype=torch.long)
		
		_times[6] += time.time() - _t
		
		for batch_index in range(batch_size):
			
			line = read_one_line()
			
			if line:
				process_one_line(line, word_ids, entity_ids, targets, batch_index)
			else:
				data_available = False
				break
		
		if data_available:
			word_ids = word_ids.view(batch_size, words_per_ent * neg_words).to(device)
			entity_ids = entity_ids.to(device)
			targets = targets.view(batch_size * words_per_ent).to(device)
			
			yield (word_ids, entity_ids), targets
			
			if current_data == canonical_words_path:
				show_progress(line_index / ent_size, title='   ')
			else:
				show_progress(line_index / get_hyperlinks_count(), title='   ')


if __name__ == '__main__':
	
	_start = time.time()
	
	batch_count = 0
	for inputs, targets in gen_minibatch():
		batch_count += 1
		print(_times)
	
	_end = time.time()
	print(f'-- calculated time is {_end - _start} sec')