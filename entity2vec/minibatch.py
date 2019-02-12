from basic.words import *
from basic.entities import *
import torch

config = configparser.ConfigParser()
file_path = os.path.dirname(__file__)
config_path = os.path.join(file_path, '../config.ini')
config.read(config_path)

canonical_words_path = config['path']['canonical']
hyp_ctx_words_path = config['path']['hyp_ctx']

batch_size = int(config['param']['batch_size'])
passes_wiki_words = int(config['param']['passes_wiki_words'])
words_per_ent = int(config['param']['words_per_ent'])
neg_words = int(config['param']['neg_words'])

current_data = canonical_words_path
current_file = open(current_data, 'r', encoding='utf8')
current_pass = 1


def read_one_line():
	global canonical_words_path, hyp_ctx_words_path, passes_wiki_words
	global current_data, current_file, current_pass

	line = current_file.readline()

	if not line:  # end of current file

		current_file.close()

		print('done!')

		if current_pass == passes_wiki_words:
			current_pass = 1
			return # ،TODO: remove this
			if current_data == canonical_words_path:
				current_data = hyp_ctx_words_path
				current_pass = 0
			else:
				return

		current_pass += 1

		print(f'-- starting {current_pass}-th pass on {current_data}.', end=' ', flush=True)

		current_file = open(current_data, 'r', encoding='utf8')
		line = current_file.readline()

	return line


def process_one_line(line, minibatch, batch_idx):

	line = line[:-1]    # removing \n at end of line

	if line.find('CANONICAL_WORDS') > 0:  # wiki canonical page words

		header, words = line.split('\tCANONICAL_WORDS')
		ent_wikiid, ent_name = header.split('\t')
		pos_word_ids = words.split('\t')

	elif line.find('LEFT_CTX') > 0:   # wiki hyperlink context word

		header, words = line.split('\tLEFT_CTX\t')
		ent_wikiid, ent_name, mention = header.split('\t')
		left_context = words.split('RIGHT_CTX')[0].split('\t')
		right_context = words.split('RIGHT_CTX')[1].split('\t')    # removing \n at end of line
		pos_word_ids = left_context + right_context

	ent_wikiid = int(ent_wikiid)
	minibatch['entity_wikiid'][batch_idx] = ent_wikiid
	minibatch['entity_id'][batch_idx] = wikiid2id[ent_wikiid]

	pos_word_ids = list(filter(lambda wid: len(wid) > 0, pos_word_ids))
	pos_word_ids = list(map(lambda wid: int(wid), pos_word_ids))

	# pos words are in vocab and not stop words from data gen step
	# use entity name if no positive context words available
	if len(pos_word_ids) == 0:
		entity_name_words = word_tokenize(ent_name)
		entity_name_words = list(filter(lambda w: w in word2id.keys(), entity_name_words))
		pos_word_ids = list(map(lambda w: word2id[w], entity_name_words))

	# get ranfom sample if still empty
	if len(pos_word_ids) == 0:
		pos_word_ids = get_unig_sample(10)

	# fill with negative words sampled from powered unigram
	neg_sample = get_unig_sample(words_per_ent * neg_words).view(words_per_ent, neg_words)
	minibatch['word_ids'][batch_idx] = neg_sample

	# sample some positive words, place them somewhere between negative words
	# and store that index in targets
	pos_sample_indexes = torch.randint(high=len(pos_word_ids), size=(words_per_ent,))
	targets = torch.randint(high=neg_words, size=(words_per_ent,))

	for w in range(words_per_ent):
		pos_id = pos_word_ids[pos_sample_indexes[w]]
		# pos_id = word2id[pos_word]
		minibatch['word_ids'][batch_idx][w][targets[w]] = pos_id

	return targets


def gen_minibatch():
	global batch_size, passes_wiki_words, words_per_ent, neg_words
	global canonical_words_path, hyp_ctx_words_path, passes_wiki_words
	global current_data, current_file, current_pass

	print(f'-- starting first pass.', end=' ', flush=True)

	current_data = canonical_words_path
	current_file = open(current_data, 'r', encoding='utf8')
	current_pass = 1

	data_available = True

	while data_available:

		minibatch = {'word_ids': torch.ones(batch_size, words_per_ent, neg_words, device=device).long().mul(unk_word_id),
		             'entity_wikiid': torch.ones(batch_size, device=device).long(),
		             'entity_id': torch.ones(batch_size, device=device).long()}

		targets = torch.zeros(batch_size, words_per_ent, device=device).long()

		for batch_index in range(batch_size):
			line = read_one_line()
			if not line:
				data_available = False
				break
			targets[batch_index] = process_one_line(line, minibatch, batch_index)

		if data_available:

			minibatch['word_ids'] = minibatch['word_ids'].view(batch_size, words_per_ent * neg_words)
			targets = targets.view(batch_size * words_per_ent)

			yield minibatch, targets


if __name__ == '__main__':

	batch_count = 0
	for inputs, targets in gen_minibatch():
		print(f'-- inputs: {inputs}')
		print(f'-- targets: {targets}')
		if batch_count > 10:
			break